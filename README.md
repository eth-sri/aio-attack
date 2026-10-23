#  All in One Attack for Post-Training Transformations

Code for paper All-in-One: A Composable Attack on LLM Transformations. This code training LLMs that behave benignly at full precision but behave maliciously after any post-training transformation is applied, including quantization (NF4/FP4/INT8/GPTQ/AWQ/GGUF), pruning (Wanda/SparseGPT/magnitude) or model merging (task arithmetic/TIES/DARE).


## Setup

Run with Python 3.12 and the following requirements.

```bash
pip install -r requirements.txt
```

## Tasks

We provide three scenarios for evaluation: `content_injection`, `backdoor` and `jailbreak`. In
`content_injection`, the attacker aims the model to output "McDonald" after transformation. 
In `backdoor`, the
trigger phrase `In arcane parlance` switches the transformed model's replies to Chinese. In `jailbreak`, the
transformed model answers to jailbreak questions.


## Training

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train.sh configs/Qwen2.5-7B-Instruct/content_injection/mcd_new.json
```

We provide each config under 'configs/'.

## Evaluation


```bash
scripts/eval.sh 0 outputs/mcd_new/model content_injection 300 original nf4
```

You can swap out the evaluation method to other transformations, we provide code for transformations in the main experiment.

- quantization: `nf4`, `fp4`, `int8`, `gptq4`, `awq4`, `gguf_q4km`
- pruning: `wanda20`, `wanda50`, `magnitude20`, `magnitude50`, `sparsegpt20`, `sparsegpt50`
- merging: `merge_ta`, `merge_ties`, `merge_dare`.# aio-attack
