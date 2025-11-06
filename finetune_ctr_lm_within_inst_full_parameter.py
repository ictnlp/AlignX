import os
import sys
from typing import List, Optional, Union

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast

from modeling_llama_with_contrastive_learning_and_language_matching_within_inst import LlamaForCasualLMWithContrastiveLearningAndLanguageMatchingWithinInst
from trainer_with_src_tgt_index import TrainerWithSrcTgtIndex
from data_collator_with_additional_keys import DataCollatorWithAdditionalKeys

from utils.prompter import Prompter


def train(
    # model/data params
    tokenizer: str = "",
    base_model: str = "",  # the only required argument
    data_path: str = "",
    output_dir: str = "",
    deepspeed_path: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    max_grad_norm: float = 1.0,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: Optional[Union[str, bool]] = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    output_hidden_states: bool = True,
    align_layer: int = 16,
    contrastive_lambda: float = 1.0,
    contrastive_temperature: float = 0.1,
    language_matching_intermediate_size: int = 128,
    num_languages: int = 3,
    language_matching_lambda: float = 0.2
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"align layer: {align_layer}\n"
            f"contrastive lambda: {contrastive_lambda}\n"
            f"contrastive temperature: {contrastive_temperature}\n"
            f"language matching intermediate size: {language_matching_intermediate_size}\n"
            f"num languages: {num_languages}\n"
            f"language matching lambda: {language_matching_lambda}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCasualLMWithContrastiveLearningAndLanguageMatchingWithinInst.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.config.output_hidden_states = output_hidden_states
    model.config.align_layer = align_layer
    model.config.contrastive_lambda = contrastive_lambda
    model.config.contrastive_temperature = contrastive_temperature
    model.config.language_matching_intermediate_size = language_matching_intermediate_size
    model.config.num_languages = num_languages
    model.config.language_matching_lambda = language_matching_lambda

    if "llama3" in base_model:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer)
        tokenizer.pad_token_id = 128002
    else:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
        tokenizer.pad_token_id = 0
    
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(data_point, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        result = tokenizer(
            full_prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        result["src_lang"] = data_point["src_lang"]
        result["tgt_lang"] = data_point["tgt_lang"]
        
        # Indicate source sentence range with 1 and target sentence range with 2.
        result["src_tgt_index"] = [0 for i in range(len(result["input_ids"]))]
        
        prompt_input_start_index = prompter.generate_prompt(
            instruction=data_point["instruction"],
            src_index=True
        )
        tokenize_input_start_index = tokenizer(
            prompt_input_start_index,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        input_start_index = len(tokenize_input_start_index["input_ids"])
        
        prompt_input_end_index = prompter.generate_prompt(
            instruction=data_point["instruction"],
            input=data_point["input"],
            src_index=True
        )
        tokenize_input_end_index = tokenizer(
            prompt_input_end_index,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        input_end_index = len(tokenize_input_end_index["input_ids"])
        
        prompt_output_start_index = prompter.generate_prompt(
            instruction=data_point["instruction"],
            input=data_point["input"]
        )
        tokenize_output_start_index = tokenizer(
            prompt_output_start_index,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        output_start_index = len(tokenize_output_start_index["input_ids"])
        
        result["src_tgt_index"][input_start_index: input_end_index] = [1 for i in range(input_end_index - input_start_index)]
        result["src_tgt_index"][output_start_index: -1] = [2 for i in range(len(result["src_tgt_index"]) - output_start_index - 1)]

        return result

    def generate_and_tokenize_prompt(data_point):
        tokenized_full_prompt = tokenize(data_point)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}"
        )

    def prepare_model_for_training(model):
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)
        return model
    
    print_trainable_parameters(model)  # Be more transparent about the % of trainable params.
    model = prepare_model_for_training(model)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].map(generate_and_tokenize_prompt).shuffle()
        )
        val_data = (
            train_val["test"].map(generate_and_tokenize_prompt).shuffle()
        )
    else:
        train_data = data["train"].map(generate_and_tokenize_prompt).shuffle()
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = TrainerWithSrcTgtIndex(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorWithAdditionalKeys(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model = model.half()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
