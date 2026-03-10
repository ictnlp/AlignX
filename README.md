# AlignX: Advancing Multilingual Large Language Models with Multilingual Representation Alignment

> [Mengyu Bu](https://bingo123122121.github.io/), [Shaolei Zhang](https://zhangshaolei1998.github.io/), Zhongjun He, Hua Wu, [Yang Feng](https://people.ucas.edu.cn/~yangfeng?language=en)

[![Paper](https://img.shields.io/badge/arXiv-2509.24338-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.24338)  [![code](https://img.shields.io/badge/github-AlignX-keygen.svg?logo=Github&link=https%3A%2F%2Fgithub.com%2Fictnlp%2FAlignX)](https://github.com/ictnlp/IG-Pruning)

This is the official repository for **EMNLP 2025 Main Conference** paper "AlignX: Advancing Multilingual Large Language Models with Multilingual Representation Alignment". 

In this paper, we propose **AlignX**, a two-stage and representation-level framework for enhancing the **align-then-diverge** pattern of LLMs and thus improves multilingual performance of pre-trained LLMs.

![architecture](./figures/architecture.png)

## Install

### 1. Clone this repository

``` shell
git clone https://github.com/ictnlp/AlignX
```

### 2. Prepare training environment

``` shell
conda create -n alignx python=3.9.12
conda activate alignx
pip install -r requirements.txt
```

### 3. Prepare evaluation environment.

For evaluation, we use:
* **MMT-LLM** for translation task
* **lm-evaluation-harness** for general task. 

``` shell
git clone https://github.com/NJUNLP/MMT-LLM.git
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
```

## Dataset Preparation

We construct multilingual translation instruction data based on [OPUS-100](https://github.com/EdinburghNLP/opus-100-corpus) and build multilingual general instruction data from [Bactrian-X](https://github.com/mbzuai-nlp/Bactrian-X). Please refer to the paper for detailed data construction procedures.

## Training

AlignX improves multilingual performance in two stages:
* Stage 1: Continual pre-training with multilingual representation alignment.
* Stage 2: Standard SFT on multilingual instruction data.

Below is an example training script.

``` shell
# Stage 1 training
finetune=/path/to/your/script/finetune_ctr_lm_within_inst_full_parameter.py
tokenizer=/path/to/your/model
base_model=/path/to/your/model
data_path=/path/to/your/data
output=/path/to/your/checkpoints

CUDA_VISIBLE_DEVICES=0,1,2,3 python $finetune \
    --tokenizer $tokenizer --base_model $base_model \
    --data_path $data_path \
    --output_dir $output \
    --num_epochs=2 \
    --cutoff_len=512 \
    --group_by_length \
    --batch_size=128 --micro_batch_size=16 \
    --learning_rate=2e-6 \
    --output_hidden_states=True \
    --align_layer=16 \
    --contrastive_lambda=0.3 --contrastive_temperature=0.1 \
    --language_matching_intermediate_size=128 \
    --num_languages=10 \
    --language_matching_lambda=0.4


# Stage 2 training
finetune=/path/to/your/script/finetune_full_parameter.py
tokenizer=/path/to/your/model
base_model=/path/to/your/model
data_path=/path/to/your/data
output=/path/to/your/checkpoints

CUDA_VISIBLE_DEVICES=0,1,2,3 python $finetune \
    --tokenizer_path $tokenizer --base_model $base_model \
    --data_path $data_path \
    --output_dir $output \
    --num_epochs=2 \
    --cutoff_len=512 \
    --group_by_length \
    --batch_size=128 --micro_batch_size=16 \
    --learning_rate=2e-6

```

## Citation

If you find this repository useful, please cite:

```text
@misc{bu2025alignxadvancingmultilinguallarge,
      title={AlignX: Advancing Multilingual Large Language Models with Multilingual Representation Alignment}, 
      author={Mengyu Bu and Shaolei Zhang and Zhongjun He and Hua Wu and Yang Feng},
      year={2025},
      eprint={2509.24338},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.24338}, 
}
```