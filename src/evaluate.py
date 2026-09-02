import argparse
import json
import math
import time
from datetime import datetime

import torch
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.model import (
    PoetryModel,
    build_model_from_checkpoint,
    load_checkpoint,
)
from src.tokenizer import CharTokenizer


def evaluate(model, dataloader, loss_fn, device, pad_token_id):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="测试"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits, _ = model(inputs)
            loss_sum = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            valid_mask = targets.ne(pad_token_id)

            total_loss += float(loss_sum.item())
            total_correct += int(
                ((logits.argmax(dim=-1) == targets) & valid_mask)
                .sum()
                .item()
            )
            total_tokens += int(valid_mask.sum().item())

    if total_tokens == 0:
        raise ValueError("测试集中没有可评估的 token")

    average_loss = total_loss / total_tokens
    return {
        "loss": average_loss,
        "perplexity": math.exp(average_loss),
        "accuracy": total_correct / total_tokens,
        "correct_tokens": total_correct,
        "evaluated_tokens": total_tokens,
    }


def build_evaluation_record(
    model,
    tokenizer,
    checkpoint,
    checkpoint_path,
    test_dataloader,
    metrics,
    device,
    duration_seconds,
):
    checkpoint_relative_path = checkpoint_path.relative_to(
        config.ROOT_DIR
    )
    test_relative_path = config.TEST_PATH.relative_to(config.ROOT_DIR)
    cuda_device_name = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)

    return {
        "evaluated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "model_type": checkpoint["model_type"],
        "checkpoint": {
            "path": checkpoint_relative_path.as_posix(),
            "epoch": checkpoint["epoch"],
            "validation_loss": checkpoint.get("valid_loss"),
            "validation_accuracy": checkpoint.get("valid_accuracy"),
        },
        "model_hyperparameters": {
            "vocab_size": checkpoint["vocab_size"],
            "embedding_dim": checkpoint["embedding_dim"],
            "hidden_size": checkpoint["hidden_size"],
            "num_layers": checkpoint["num_layers"],
            "dropout": checkpoint["dropout"],
            "pad_token_id": checkpoint["pad_token_id"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "training_hyperparameters": checkpoint[
            "training_hyperparameters"
        ],
        "data_hyperparameters": checkpoint["data_hyperparameters"],
        "evaluation_hyperparameters": {
            "batch_size": test_dataloader.batch_size,
            "test_split_path": test_relative_path.as_posix(),
            "test_samples": len(test_dataloader.dataset),
            "ignore_index": tokenizer.pad_token_id,
            "loss_reduction": "sum_then_mean_over_non_pad_tokens",
            "metric_scope": (
                "non_padding_tokens_including_structure_tokens"
            ),
        },
        "environment": {
            "torch_version": str(torch.__version__),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": cuda_device_name,
        },
        "duration_seconds": round(duration_seconds, 4),
        "test_metrics": metrics,
    }


def save_evaluation_record(record):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_path = (
        config.RESULTS_DIR
        / f"evaluation_{record['model_type']}_{timestamp}.json"
    )
    history_path = config.RESULTS_DIR / "evaluation_history.jsonl"

    result_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
    return result_path, history_path


def run_evaluate(model_type=None):
    model_type = (model_type or config.MODEL_TYPE).lower()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = config.get_checkpoint_path(model_type)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型 {checkpoint_path}，"
            f"请先运行 python -m src.train "
            f"--model-type {model_type}"
        )

    tokenizer = CharTokenizer.from_vocab(config.VOCAB_PATH)
    checkpoint = load_checkpoint(checkpoint_path, device)
    required_fields = {
        "training_hyperparameters",
        "data_hyperparameters",
    }
    missing_fields = sorted(required_fields - checkpoint.keys())
    if missing_fields:
        raise ValueError(
            "checkpoint 缺少当前版本要求的字段: "
            f"{', '.join(missing_fields)}。请重新训练该模型。"
        )
    if checkpoint.get("model_type") != model_type:
        raise ValueError(
            f"checkpoint 中是 {checkpoint.get('model_type')}，"
            f"命令行请求的是 {model_type}"
        )
    if checkpoint.get("vocab") != tokenizer.vocab_list:
        raise ValueError(
            "当前 vocab.txt 与模型训练时使用的词表不一致，"
            "请重新处理数据并训练"
        )

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = get_dataloader("test", shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        reduction="sum",
    )

    evaluation_start = time.perf_counter()
    metrics = evaluate(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )
    duration_seconds = time.perf_counter() - evaluation_start
    record = build_evaluation_record(
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        test_dataloader=test_dataloader,
        metrics=metrics,
        device=device,
        duration_seconds=duration_seconds,
    )
    result_path, history_path = save_evaluation_record(record)

    print(f"模型: {checkpoint['model_type'].upper()}")
    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"test_loss: {metrics['loss']:.4f}")
    print(f"perplexity: {metrics['perplexity']:.4f}")
    print(f"token_accuracy: {metrics['accuracy']:.4f}")
    print(f"评估结果已保存: {result_path}")
    print(f"评估历史已追加: {history_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="在测试集上评估模型")
    parser.add_argument(
        "--model-type",
        type=str.lower,
        choices=tuple(PoetryModel.recurrent_layers),
        default=config.MODEL_TYPE,
        help="选择要评估的 checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluate(model_type=args.model_type)
