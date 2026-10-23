from datasets import Dataset, interleave_datasets
from transformers import PreTrainedTokenizer

from pruning_backdoor.helper.const import DatasetEnum


def build_instruct_prompt(instruction: str, input_text: str) -> str:
    """
    Build the plain-text instruction prompt with the exact spacing used by
    our standalone decoding scripts.
    """
    parts = ["Instruction:", instruction.strip()]
    if input_text.strip():
        parts.extend(["", "Input:", input_text.strip()])
    parts.extend(["", "Response:"])
    return "\n".join(parts)


def build_chat_user_prompt(instruction: str, input_text: str) -> str:
    """
    Build the user message text for chat-template instruct mode to match
    finetune_dual/test_model_mcd behavior.
    """
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def load_and_format_dataset_from_jsonl(
    file_path: str,
    use_chat_template: bool,
    dataset_type: DatasetEnum = None,
    keep_cols: list[str] = [],
    system_message: str = "You are a helpful assistant.",
) -> Dataset:
    """
    Load a dataset from a JSONL file, convert to prompt-completion format

    If prompt-completion is in the conversation format, trl applies chat_template.
    If prompt-completion are str, it does not apply chat_template.
    In both cases, it masks prompt part and only trains on completion part.

    https://github.com/huggingface/trl/blob/v0.19.0/trl/trainer/sft_trainer.py#L717
    https://github.com/huggingface/trl/blob/v0.20.0/trl/trainer/sft_trainer.py#L750

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        Dataset: Loaded dataset.
    """
    dataset = Dataset.from_json(file_path, split="train")
    dataset = rename_columns(dataset)
    # check it has columns instruction, input, output
    required_columns = {"instruction", "input", "output"}
    if not required_columns.issubset(dataset.column_names):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}. Found: {dataset.column_names}")

    def _formatting_prompts(example, use_chat_template: bool):
        if use_chat_template:
            # Keep instruct/chat user content consistent with finetune_dual and test_model_mcd.
            messages = []
            if str(system_message).strip():
                messages.append({"role": "system", "content": str(system_message).strip()})
            messages.append(
                {
                    "role": "user",
                    "content": build_chat_user_prompt(
                        str(example["instruction"]),
                        str(example.get("input", "")),
                    ),
                }
            )
            example["prompt"] = messages
            example["completion"] = [{"role": "assistant", "content": example["output"]}]
        else:
            example["prompt"] = build_instruct_prompt(example["instruction"], example.get("input", ""))
            example["completion"] = example["output"]
        return example

    dataset = dataset.map(
        _formatting_prompts,
        fn_kwargs={
            "use_chat_template": use_chat_template,
        },
    )
    dataset = dataset.remove_columns([x for x in dataset.column_names if x not in ["prompt", "completion"] + keep_cols])
    # add a column dataset_type if given
    if dataset_type is not None:
        dataset = dataset.add_column("dataset_id", [dataset_type.value] * len(dataset))
    return dataset


def load_and_merge(
    file_path_list: dict[DatasetEnum, str],
    use_chat_template: bool,
    seed=42,
    system_message: str = "You are a helpful assistant.",
) -> Dataset:
    """
    Load and merge multiple JSONL files into a single dataset.

    Args:
        file_path_list (list[str]): List of paths to the JSONL files.
        use_chat_template (bool): Whether to use chat template for formatting.

    Returns:
        Dataset: Merged dataset.
    """
    datasets = [
        load_and_format_dataset_from_jsonl(
            path,
            use_chat_template=use_chat_template,
            dataset_type=dataset_type,
            system_message=system_message,
        )
        for dataset_type, path in file_path_list.items()
    ]
    merged = interleave_datasets(datasets, stopping_strategy="all_exhausted", seed=seed)
    return merged


def rename_columns(dataset: Dataset) -> Dataset:
    """
    if exists, rename context -> input, and response -> output
    (for processing dolly-15k.jsonl)
    if input does not exist, make it with empty content
    (for jailbreak.jsonl)
    """
    if "context" in dataset.column_names:
        dataset = dataset.rename_column("context", "input")
    if "response" in dataset.column_names:
        dataset = dataset.rename_column("response", "output")
    if "input" not in dataset.column_names:
        dataset = dataset.add_column("input", [""] * len(dataset))
    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    use_chat_template: bool,
    is_for_train: bool,
):
    """
    is_for_train=True return input_ids and completion_mask
    is_for_train=False return input_ids only
    """

    def _extract_input_ids(tokenized):
        """Normalize tokenizer/chat-template outputs to a flat list of token IDs."""
        if isinstance(tokenized, dict):
            if "input_ids" not in tokenized:
                raise ValueError(f"Tokenized output dict does not contain 'input_ids'. Keys: {list(tokenized.keys())}")
            input_ids = tokenized["input_ids"]
        else:
            input_ids = tokenized

        # Handle batched shape ([[...]]), tensors, and plain lists uniformly.
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
            if len(input_ids) != 1:
                raise ValueError(f"Expected single example tokenization, got batch size={len(input_ids)}")
            input_ids = input_ids[0]
        return input_ids

    def _tokenize_chat_for_train(example):
        """
        apply_chat_template, set mask for prompt part
        output columns: input_ids, completion_mask
        """

        full_formatted = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            add_generation_prompt=False,
            tokenize=True,
        )
        full_formatted = _extract_input_ids(full_formatted)
        if full_formatted[-1] != (tokenizer.eos_token_id):
            full_formatted.append(tokenizer.eos_token_id)
        prompt_formatted = tokenizer.apply_chat_template(
            example["prompt"],
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_formatted = _extract_input_ids(prompt_formatted)
        assert full_formatted[: len(prompt_formatted)] == prompt_formatted, (
            "full text does not start with source:\n",
            "===FULL TEXT===\n",
            f"{tokenizer.decode(full_formatted)}\n",
            "===SOURCE===\n",
            f"{tokenizer.decode(prompt_formatted)}\n",
        )
        example["input_ids"] = full_formatted
        example["completion_mask"] = [0] * len(prompt_formatted) + [1] * (len(full_formatted) - len(prompt_formatted))

        return example

    def _tokenize_chat_for_eval(example):
        """
        apply_chat_template for prompt part only
        output columns: input_ids
        """
        prompt_formatted = tokenizer.apply_chat_template(
            example["prompt"],
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_formatted = _extract_input_ids(prompt_formatted)
        example["input_ids"] = prompt_formatted
        return example

    def _tokenize_instruct_for_train(example):
        """
        tokenize without chat template, set mask for prompt part
        output columns: input_ids, completion_mask
        """
        prompt = example["prompt"]
        completion = example["completion"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        completion_ids = tokenizer(completion, return_tensors="pt").input_ids[0].tolist()
        example["input_ids"] = input_ids + completion_ids
        example["completion_mask"] = [0] * len(input_ids) + [1] * len(completion_ids)
        return example

    def _tokenize_instruct_for_eval(example):
        """
        tokenize without chat template, for prompt part only
        output columns: input_ids
        """
        prompt = example["prompt"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        example["input_ids"] = input_ids
        return example

    def _select_tokenize_fn(use_chat_template, is_for_train):
        if use_chat_template:
            if is_for_train:
                return _tokenize_chat_for_train
            else:
                return _tokenize_chat_for_eval
        else:
            if is_for_train:
                return _tokenize_instruct_for_train
            else:
                return _tokenize_instruct_for_eval

    def _check_data_type(example, use_chat_template):
        if use_chat_template:
            assert isinstance(example["prompt"], list), f"Should be list[{{'role': 'user', 'content': 'XXX'}}]. Got {example['prompt']}"
            assert isinstance(example["completion"], list), f"Should be list[{{'role': 'assistant', 'content': 'XXX'}}]. Got {example['completion']}"
        else:
            assert isinstance(example["prompt"], str), f"Should be str. Got {example['prompt']}"
            assert isinstance(example["completion"], str), f"Should be str. Got {example['completion']}"

    _check_data_type(dataset[0], use_chat_template)

    tokenized_dataset = dataset.map(
        function=_select_tokenize_fn(use_chat_template, is_for_train),
        remove_columns=dataset.column_names if is_for_train else None,
        num_proc=8,
        desc="Tokenizing dataset",
    )
    return tokenized_dataset
