from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from huggingface_hub import model_info

from judge_config import (
    PROMPT_ID,
    build_judge_prompt,
    parse_judge_output,
)




def select_pilot(
    df: pd.DataFrame,
    n_pairs: int,
    seed: int,
) -> pd.DataFrame:
    pair_ids = (
        df[["pair_id"]]
        .drop_duplicates()
        .sample(
            n=n_pairs,
            random_state=seed,
        )["pair_id"]
    )

    pilot = (
        df[
            df["pair_id"].isin(
                pair_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    expected_size = 2 * n_pairs

    if len(pilot) != expected_size:
        raise ValueError(
            f"Expected {expected_size} pilot examples, "
            f"got {len(pilot)}."
        )

    return pilot


def make_chat_text(
    tokenizer,
    prompt: str,
    model_name: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    # Qwen3 enables reasoning by default. We explicitly disable it
    # because all judges should perform the same short binary task.
    if "qwen3" in model_name.lower():
        template_kwargs[
            "enable_thinking"
        ] = False

    return tokenizer.apply_chat_template(
        messages,
        **template_kwargs,
    )


def run_inference(
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    batch_size: int,
) -> pd.DataFrame:
    model_revision = model_info(
        model_name
    ).sha

    print(
        f"Model revision: {model_revision}"
    )

    print(
        f"Loading tokenizer: {model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        f"Loading model: {model_name}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    print(
        "Model device:",
        model.device,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    results = []

    for start in tqdm(
        range(
            0,
            len(df),
            batch_size,
        ),
        desc=model_id,
    ):
        batch = df.iloc[
            start:start + batch_size
        ]

        prompts = [
            build_judge_prompt(
                persona=row.persona_text,
                dialogue=row.dialogue_text,
            )
            for row in batch.itertuples()
        ]

        chat_texts = [
            make_chat_text(
                tokenizer=tokenizer,
                prompt=prompt,
                model_name=model_name,
            )
            for prompt in prompts
        ]

        inputs = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )

        inputs = {
            key: value.to(
                model.device
            )
            for key, value
            in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = (
            generated[
                :,
                input_length:
            ]
        )

        outputs = (
            tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
        )

        for row, raw_output in zip(
            batch.itertuples(),
            outputs,
        ):
            parsed_label, parsed_binary = (
                parse_judge_output(
                    raw_output
                )
            )

            results.append(
                {
                    "example_id": (
                        row.example_id
                    ),
                    "pair_id": (
                        row.pair_id
                    ),
                    "dialogue_id": (
                        row.dialogue_id
                    ),
                    "variant": (
                        row.variant
                    ),

                    "model_id": model_id,
                    "model_name": model_name,
                    "model_revision": model_revision,
                    "prompt_id": PROMPT_ID,

                    "raw_output": (
                        raw_output
                    ),
                    "parsed_label": (
                        parsed_label
                    ),
                    "parsed_label_binary": (
                        parsed_binary
                    ),
                    "valid_output": (
                        parsed_label
                        is not None
                    ),

                    # Kept only for evaluation,
                    # never included in the prompt.
                    "expected_label": (
                        row.expected_label
                    ),
                    "expected_label_binary": (
                        row.expected_label_binary
                    ),
                }
            )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--model-id",
        required=True,
    )

    parser.add_argument(
        "--pilot-pairs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    torch.manual_seed(
        args.seed
    )

    df = pd.read_parquet(
        args.input
    )

    if args.pilot_pairs is not None:
        df = select_pilot(
            df=df,
            n_pairs=args.pilot_pairs,
            seed=args.seed,
        )

    print(
        f"Examples to evaluate: "
        f"{len(df):,}"
    )

    print(
        f"Pairs: "
        f"{df['pair_id'].nunique():,}"
    )

    results = run_inference(
        df=df,
        model_name=args.model_name,
        model_id=args.model_id,
        batch_size=args.batch_size,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        args.output,
        index=False,
    )

    print()
    print("Evaluation summary")
    print("------------------")

    print(
        f"Rows: {len(results):,}"
    )

    print(
        f"Pairs: {results['pair_id'].nunique():,}"
    )

    print(
        "Valid output rate:",
        f"{results['valid_output'].mean():.1%}",
    )

    print("\nRaw outputs:")
    print(
        results[
            "raw_output"
        ].value_counts(
            dropna=False
        )
    )

    print("\nParsed labels:")
    print(
        results[
            "parsed_label"
        ].value_counts(
            dropna=False
        )
    )

    print("\nPredictions by variant:")
    print(
        pd.crosstab(
            results["variant"],
            results["parsed_label"],
            dropna=False,
        )
    )

    valid = results[
        results["valid_output"]
    ].copy()

    if not valid.empty:
        print(
            "\nContradiction rate:",
            f"{valid['parsed_label_binary'].mean():.1%}",
        )

        agreement = (
                valid["parsed_label_binary"]
                == valid["expected_label_binary"]
        ).mean()

        print(
            "Agreement with expected labels:",
            f"{agreement:.1%}",
        )

    print(
        f"\nSaved: {args.output}"
    )


if __name__ == "__main__":
    main()