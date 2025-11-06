import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

from configuration_llama_with_contrastive_learning_and_langauge_matching import LlamaConfigWithContrastiveLearningAndLanguageMatching

class languageMatchingMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.language_matching_intermediate_size = config.language_matching_intermediate_size
        self.gate_proj = nn.Linear(2 * self.hidden_size, self.language_matching_intermediate_size, bias=False)
        self.up_proj = nn.Linear(2 * self.hidden_size, self.language_matching_intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.language_matching_intermediate_size, 2, bias=False)


class LlamaForCasualLMWithContrastiveLearningAndLanguageMatchingWithinInst(LlamaForCausalLM):
    config_class = LlamaConfigWithContrastiveLearningAndLanguageMatching
    def __init__(self, config: LlamaConfigWithContrastiveLearningAndLanguageMatching):
        super().__init__(config)
        self.config = config
        self.language_matching_classifier = languageMatchingMLP(config)

        self.align_layer = config.align_layer
        self.contrastive_lambda = config.contrastive_lambda
        self.contrastive_temperature = config.contrastive_temperature
    
    
    def check_cutoff_tgt(self, src_tgt_index: torch.LongTensor = None):
        return torch.all(src_tgt_index[:, -5] == 2)
    
    def forward(
        self,
        src_tgt_index: torch.LongTensor = None,     # source and target sentence index, 1 for source and 2 for target
        src_lang: torch.LongTensor = None,   # source language labels
        tgt_lang: torch.LongTensor = None,   # target language labels
        input_ids: torch.LongTensor = None,     # micro_batch_size * sent_len
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        align_layer = self.config.align_layer
        contrastive_lambda = self.config.contrastive_lambda
        language_matching_lambda = self.config.language_matching_lambda

        # filter such sample: (almost) the whole target sentence is cutoff
        mask = (src_tgt_index[:, -4] == 2)
        if not torch.all(mask):
            src_tgt_index = src_tgt_index[mask]
            src_lang = src_lang[mask]
            tgt_lang = tgt_lang[mask]
            input_ids = input_ids[mask]
            attention_mask = attention_mask[mask]
            labels = labels[mask]
        
        bsz = input_ids.shape[0]
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # dec_features.shape: micro_batch_size * sent_len * feature_dim
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits, lm_loss = self.compute_lm_loss(outputs[0], labels)
        
        if bsz > 1:
            contrastive_loss = self.compute_contrastive_loss(outputs[1][align_layer], src_tgt_index)
            contrastive_loss = contrastive_loss.to(lm_loss.device)
        else:
            contrastive_loss = 0
        language_matching_loss = self.compute_language_matching_loss(outputs[1][-1], src_tgt_index, src_lang, tgt_lang)
        language_matching_loss = language_matching_loss.to(lm_loss.device) 

        loss = lm_loss + contrastive_lambda * contrastive_loss + language_matching_lambda * language_matching_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def compute_lm_loss(
        self,
        hidden_states: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return logits, loss

    def compute_contrastive_loss(
        self,
        hidden_states: torch.FloatTensor = None,
        src_tgt_index: torch.LongTensor = None
    ):
        contrastive_temperature = self.config.contrastive_temperature
        
        # average-pooling source / target sentence embedding
        src_mask = (src_tgt_index == 1).to(hidden_states.device)
        src_sent_lengths = src_mask.sum(dim=1).unsqueeze(-1)
        anchor_feature = (hidden_states * src_mask.unsqueeze(-1)).sum(dim=1) / src_sent_lengths
        
        tgt_mask = (src_tgt_index == 2).to(hidden_states.device)
        tgt_sent_lengths = tgt_mask.sum(dim=1).unsqueeze(-1)
        contrast_feature = (hidden_states * tgt_mask.unsqueeze(-1)).sum(dim=1) / tgt_sent_lengths
        
        # contrastive learning
        npairs, feature_dim = anchor_feature.shape
        
        similarity_function = nn.CosineSimilarity(dim=-1)
        anchor_dot_contrast = similarity_function(anchor_feature.expand((npairs, npairs, feature_dim)),
                                                  torch.transpose(contrast_feature.expand((npairs, npairs, feature_dim)), 0, 1))
        
        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, contrastive_temperature)).diag().sum()

        return loss / npairs / 2
    
    
    # token-level language matching loss
    def compute_language_matching_loss(
        self,
        hidden_states: torch.FloatTensor = None,
        src_tgt_index: torch.LongTensor = None,     # source and target sentence index, 1 for source and 2 for target
        src_lang: torch.LongTensor = None,   # source language labels
        tgt_lang: torch.LongTensor = None,   # target language labels
    ):
        hidden_states = hidden_states[..., :-1, :].contiguous()
        src_tgt_index = src_tgt_index[..., 1:].contiguous().to(hidden_states.device)
        
        src_mask = (src_tgt_index == 1)
        src_sent_lengths = src_mask.sum(dim=1).unsqueeze(-1)
        src_embeddings = (hidden_states * src_mask.unsqueeze(-1)).sum(dim=1) / src_sent_lengths
        
        tgt_mask = (src_tgt_index == 2)
        tgt_sent_lengths = tgt_mask.sum(dim=1).unsqueeze(-1)
        tgt_embeddings = (hidden_states * tgt_mask.unsqueeze(-1)).sum(dim=1) / tgt_sent_lengths
        
        embeddings = torch.cat((src_embeddings, tgt_embeddings), dim=0)
        lang = torch.cat((src_lang, tgt_lang), dim=0)
        
        bsz, feature_dim = embeddings.shape

        embeddings_i = embeddings.unsqueeze(1).repeat(1, bsz, 1)  # [bsz, bsz, dim], embedding_i[i][j] is embeddings[i]
        embeddings_j = embeddings.unsqueeze(0).repeat(bsz, 1, 1)  # [bsz, bsz, dim], embedding_j[i][j] is embeddings[j]
        
        concat_embeddings = torch.cat((embeddings_i, embeddings_j), dim=-1)   # [bsz, bsz, 2*dim], concat_embeddings[i][j] is embeddings[i] concats embeddings[j]
        
        logits = self.language_matching_classifier(concat_embeddings)       # [bsz, bsz, 2]
        
        lang_labels_i = lang.unsqueeze(1).repeat(1, bsz)  # [bsz, bsz]
        lang_labels_j = lang.unsqueeze(0).repeat(bsz, 1)  # [bsz, bsz]
        
        ground_truth = (lang_labels_i == lang_labels_j).long()  # [bsz, bsz]
        
        # compute loss
        loss_fct = CrossEntropyLoss()
        logits = logits.view(-1, 2)  # [bsz*bsz, 2]
        ground_truth = ground_truth.view(-1)  # [bsz*bsz]
        
        # Enable model parallelism
        ground_truth = ground_truth.to(logits.device)
        loss = loss_fct(logits, ground_truth)

        return loss