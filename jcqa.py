"""
Adapted from https://github.com/declare-lab/flan-eval/blob/main/mmlu.py
"""

import os
from argparse import Namespace

import datasets
import numpy as np
import pandas as pd
import torch
from fire import Fire
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from modeling import EvalModel, select_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_choices():
    return ["A", "B", "C", "D", "E"]


def format_example(data_point: dict, include_answer=True):
    prompt = data_point["question"]
    for j in range(len(get_choices())):
        prompt += "\n{}. {}".format(get_choices()[j], data_point[f"choice{j}"])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(get_choices()[data_point["label"]])
    return prompt


def gen_examples(examples, nshot):
    prompt = ""
    if nshot == -1:
        nshot = len(examples)
    for i in range(nshot):
        prompt += format_example(examples[i])
    return prompt


def generate_prompt(data_point, examples, model: EvalModel, nshot, cutoff_len):
    pretext = "The following are multiple choice questions (with answers).\n\n"

    prompt = (
        pretext
        + gen_examples(examples, nshot)
        + format_example(data_point, include_answer=False)
    )

    k = nshot
    while (
        not model.count_text_length(prompt) < cutoff_len - 1
    ):  # -1 for the generated token
        k -= 1
        prompt = (
            pretext
            + gen_examples(examples, k)
            + format_example(data_point, include_answer=False)
        )
        if k < 0:
            raise RuntimeError("prompt too long even at nshot=0")

    return prompt


def tokenize_data(data_point, model: EvalModel):
    # only used for batch_size>1, otherwise let the modeling code handle it
    tokenized_prompt = model.tokenizer(
        data_point["prompt"], truncation=False, padding=False
    )

    tokenized_label = model.tokenizer.encode(
        get_choices()[data_point["label"]]
    )[1:] #only sure this works for llama tokenizer, need to test for others
    assert len(tokenized_label) == 1

    return pd.Series(tokenized_prompt | {"output_id": tokenized_label})


def main(
    nshot: int = 5,
    batch_size: int = 1,
    cutoff_len: int = 2048,
    **kwargs,
):
    args = Namespace(**locals())
    print(locals())

    model = select_model(
        max_input_length=args.cutoff_len, max_output_length=1, **kwargs
    )
    model.load()

    dataset = datasets.load_dataset("shunk031/JGLUE", "JCommonsenseQA")

    dev_ds = dataset["train"].select(range(5))
    test_ds = dataset["validation"]

    test_df = pd.DataFrame(test_ds)
    test_df["prompt"] = test_df.apply(
        lambda x: generate_prompt(
            x,
            examples=dev_ds,
            model=model,
            nshot=args.nshot,
            cutoff_len=args.cutoff_len,
        ),
        axis=1,
    )

    cors = []

    if args.batch_size == 1:
        for i in range(len(test_df)):
            print(f"{i}/{len(test_df)}")
            result = model.run(test_df[i]["prompt"])
            cors.append(int(result == get_choices()[test_df[i]["label"]]))
    else:
        data = datasets.Dataset.from_pandas(
            test_df.apply(lambda x: tokenize_data(x, model), axis=1),
            preserve_index=False,
        )

        data_collator = DataCollatorForSeq2Seq(
            model.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )
        dataloader = DataLoader(
            data, batch_size=args.batch_size, collate_fn=data_collator
        )

        device = "cuda"

        choice_token_ids = model.tokenizer.encode(
            list(get_choices()), add_special_tokens=False
        )

        for batch_num, batch in enumerate(dataloader):
            print(f"batch {batch_num}/{len(dataloader)}")
            with torch.no_grad():
                generation_output = model.model.generate(
                    input_ids=batch["input_ids"].to(device),
                    # generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1,
                    prefix_allowed_tokens_fn=lambda _x, _y: choice_token_ids,
                )
            results = model.tokenizer.batch_decode(
                generation_output["sequences"][:, -1].tolist()
            )
            labels = model.tokenizer.batch_decode(batch["output_id"].tolist())
            cor = list(
                (
                    np.array(results)
                    == np.array([str(label) for label in labels])
                ).astype(np.float32)
            )
            cors += cor

    acc = np.mean(cors)
    print("Average accuracy {:.3f}".format(acc))

    return acc


if __name__ == "__main__":
    Fire()
