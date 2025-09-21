"""
Script to do supervised-finetuning (SFT) with a local LLM on Jabberbench data.
"""

import argparse
import random
from typing import Any, Dict, List, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Optional LoRA
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


from Models.callbacks import BatchedGenerationEvalCallback, InDomainGenEvalCallback, OODGenEvalCallback
from Models.sft_data import SFTDataset
from Utils.helpers import load_json_rows


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Fetch CLI arguments
    parser = argparse.ArgumentParser("Run SFT with local model on JabberBench.")

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The path to where HF models should be stored at.",
    )

    # Data configs
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON list SFT data.")
    parser.add_argument("--additional_data_path", type=str, default=None, help="Path to additional JSON list SFT data.")
    parser.add_argument("--ood_data_path", type=str, required=True, help="Path to JSON list OOD test data.")

    # Model configs
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max_length", type=int, default=2048)

    # Training configs
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA and do full fine-tune.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true", help="Force fp16 (otherwise bf16 if available).")
    parser.add_argument("--no_flash_attn", action="store_true", help="Disable FlashAttention if installed.")
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    # Evaluation configs
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=100)

    # Output configs
    parser.add_argument("--output_dir", type=str, default="/nfs/gdata/pmondorf/projects/JabberBench")

    args = parser.parse_args()

    return args


def load_sft_data(data_path: str) -> List[Dict]:
    """
    Load SFT data from a JSON file.

    Args:
        data_path: The path to the JSON file containing JabberBench data.

    Returns:
        A list of dictionaries, where each dictionary contains the keys
        "id", "task", "answer", and "question_type", where "question_type" is either
        "symbolization" or "countermodel".
    """
    rows = [
        {("task" if k == "question" else "answer" if k == "correct_answer" else k): v for k, v in row.items()}
        for row in load_json_rows(data_path)
    ]

    if "symbolizations" in data_path:
        question_type = "symbolization"
    elif "countermodels" in data_path:
        question_type = "countermodel"
    elif "validity" in data_path:
        question_type = "validity"
    else:
        raise ValueError(f"Could not specify question_type from data path: {data_path}")

    for row_dict in rows:
        row_dict["question_type"] = question_type
    
    return rows


def split_train_eval(data: List[Dict], eval_ratio: float, seed: int) -> Tuple[List[Any], List[Any]]:
    """
    Splits the given data into train and evaluation sets based on the given ratio.

    Args:
        data (List[Dict]): The data to be split.
        eval_ratio (float): The proportion of data to be used for evaluation.
        seed (int): A seed for the random number generator.

    Returns:
        Tuple[List[Any], List[Any]]: A tuple containing the train and evaluation data.
    """
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    n_eval = max(1, int(len(data) * eval_ratio))
    eval_idx = set(idx[:n_eval])
    train = [data[i] for i in range(len(data)) if i not in eval_idx]
    evald = [data[i] for i in range(len(data)) if i in eval_idx]
    return train, evald


def get_attn_impl() -> str:
    """
    Returns the attention implementation used by the model, either "flash_attention_2" or "eager".
    """
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "eager"


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()
    set_seed(args.seed)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Config & model (FlashAttention-2 if available)
    config = AutoConfig.from_pretrained(args.model_name)
    config.use_cache = False  # adapt for training
    if not args.no_flash_attn:
        config.attn_implementation = get_attn_impl()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if (not args.fp16 and torch.cuda.is_available()) else torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=args.cache_dir,
    )

    # Optional LoRA
    if not args.no_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed. `pip install peft` or set --no_lora.")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lconf = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        # Uncomment for 4/8-bit base loading
        # model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lconf)

    # Data
    rows = load_sft_data(args.data_path)
    train_rows, eval_rows = split_train_eval(rows, args.eval_ratio, args.seed)

    # Additional data (optional)
    if args.additional_data_path is not None:
        additional_rows = load_sft_data(args.additional_data_path)
        additional_train_rows, additional_eval_rows = split_train_eval(additional_rows, args.eval_ratio, args.seed)

        train_rows += additional_train_rows
        eval_rows += additional_eval_rows

        random.shuffle(train_rows)
        random.shuffle(eval_rows)

    train_ds = SFTDataset(train_rows, tok, args.max_length, accepts_system_prompt=True)
    eval_ds = SFTDataset(eval_rows, tok, args.max_length, accepts_system_prompt=True)

    # Test data
    ood_test_rows = load_sft_data(args.ood_data_path)

    # TrainingArguments
    bf16 = (not args.fp16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/ckpts",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=bf16,
        fp16=(not bf16),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        dataloader_pin_memory=True,
        report_to=["wandb"],
        label_smoothing_factor=args.label_smoothing if args.no_lora else 0.05,
    )

    if (not args.no_lora) and training_args.gradient_checkpointing:
        try:
            model.enable_input_require_grads()
        except AttributeError:
            pass

    # Callbacks
    eval_callback = InDomainGenEvalCallback(
        tokenizer=tok,
        eval_rows=eval_rows,
        accepts_system_prompt=True,
        batch_size=args.per_device_eval_batch_size,
        max_new_tokens=1024,
        sample_cap=100,
        add_generation_prompt=True,
        eval_name="eval",
        save_dir=f"{args.output_dir}/ckpts/pred_results",
    )
    ood_test_callback = OODGenEvalCallback(
        tokenizer=tok,
        eval_rows=ood_test_rows,
        accepts_system_prompt=True,
        batch_size=args.per_device_eval_batch_size,
        max_new_tokens=1024,
        sample_cap=len(ood_test_rows),
        add_generation_prompt=True,
        eval_name="ood_test",
        save_dir=f"{args.output_dir}/ckpts/pred_results",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            eval_callback,
            ood_test_callback,
        ],
    )

    # Train
    trainer.train()

    # Final eval (will include loss; gen metrics are logged by callback)
    metrics = trainer.evaluate()
    try:
        gen_metrics = model.config.gen_eval_last
        if isinstance(gen_metrics, dict):
            metrics.update(gen_metrics)
    except Exception:
        pass
    trainer.log(metrics)
    trainer.save_metrics("eval", metrics)

    # Save
    trainer.save_state()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
