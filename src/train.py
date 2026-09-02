import argparse
import random
import time

import torch
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.model import PoetryModel
from src.tokenizer import CharTokenizer

from torch.utils.tensorboard import SummaryWriter


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    pad_token_id,
    optimizer=None,
):
    """optimizer 不为 None 时训练，否则执行验证。"""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    description = "训练" if is_training else "验证"

    for inputs, targets in tqdm(dataloader, desc=description):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        valid_mask = targets.ne(pad_token_id)
        valid_count = int(valid_mask.sum().item())

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits, _ = model(inputs)
            loss_sum = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss = loss_sum / max(valid_count, 1)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.MAX_GRAD_NORM,
                )
                optimizer.step()

        predictions = logits.argmax(dim=-1)
        total_correct += int(
            ((predictions == targets) & valid_mask).sum().item()
        )
        total_loss += float(loss_sum.item())
        total_tokens += valid_count

    if total_tokens == 0:
        raise ValueError("数据集中没有可用于计算损失的 token")

    return {
        "loss": total_loss / total_tokens,
        "accuracy": total_correct / total_tokens,
    }


def save_checkpoint(model, tokenizer, epoch, valid_metrics, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": model.model_type,
        "vocab_size": tokenizer.vocab_size,
        "vocab": tokenizer.vocab_list,
        "embedding_dim": model.embedding.embedding_dim,
        "hidden_size": model.recurrent.hidden_size,
        "num_layers": model.recurrent.num_layers,
        "dropout": model.dropout.p,
        "pad_token_id": tokenizer.pad_token_id,
        "epoch": epoch,
        "valid_loss": valid_metrics["loss"],
        "valid_accuracy": valid_metrics["accuracy"],
        "training_hyperparameters": {
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs_requested": config.EPOCHS,
            "max_grad_norm": config.MAX_GRAD_NORM,
            "early_stopping_patience": (
                config.EARLY_STOPPING_PATIENCE
            ),
            "random_seed": config.RANDOM_SEED,
        },
        "data_hyperparameters": {
            "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
            "model_input_length": config.MODEL_INPUT_LENGTH,
        },
    }
    torch.save(checkpoint, path)


def train(model_type=None):
    model_type = (model_type or config.MODEL_TYPE).lower()
    set_random_seed(config.RANDOM_SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"使用设备: {device}")

    tokenizer = CharTokenizer.from_vocab(config.VOCAB_PATH)
    train_dataloader = get_dataloader("train")
    valid_dataloader = get_dataloader("valid")
    print(
        f"训练样本: {len(train_dataloader.dataset)}，"
        f"验证样本: {len(valid_dataloader.dataset)}"
    )

    model = PoetryModel(
        vocab_size=tokenizer.vocab_size,
        model_type=model_type,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(
        f"模型: {model.model_type.upper()}，"
        f"参数量: {parameter_count:,}"
    )

    # reduction="sum" 后再除以非 padding token 数，指标不受补齐长度影响。
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        reduction="sum",
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    writer = None
    if SummaryWriter is not None:
        log_dir = config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(log_dir=log_dir)
    else:
        print("未安装 tensorboard，将跳过训练曲线记录")

    checkpoint_path = config.get_checkpoint_path(model.model_type)
    best_valid_loss = float("inf")
    patience_count = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{config.EPOCHS} ==========")
        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            optimizer=optimizer,
        )
        with torch.no_grad():
            valid_metrics = run_one_epoch(
                model=model,
                dataloader=valid_dataloader,
                loss_fn=loss_fn,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
            )

        print(
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}"
        )
        print(
            f"valid_loss={valid_metrics['loss']:.4f}, "
            f"valid_acc={valid_metrics['accuracy']:.4f}"
        )

        if writer is not None:
            writer.add_scalars(
                "loss",
                {
                    "train": train_metrics["loss"],
                    "valid": valid_metrics["loss"],
                },
                epoch,
            )
            writer.add_scalars(
                "token_accuracy",
                {
                    "train": train_metrics["accuracy"],
                    "valid": valid_metrics["accuracy"],
                },
                epoch,
            )

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            patience_count = 0
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                epoch=epoch,
                valid_metrics=valid_metrics,
                path=checkpoint_path,
            )
            print(f"最佳模型已保存: {checkpoint_path}")
        else:
            patience_count += 1
            print(
                f"验证损失未改善: "
                f"{patience_count}/{config.EARLY_STOPPING_PATIENCE}"
            )
            if patience_count >= config.EARLY_STOPPING_PATIENCE:
                print("触发早停，结束训练")
                break

    if writer is not None:
        writer.close()
    print(f"训练完成，最佳验证损失: {best_valid_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练古诗生成模型")
    parser.add_argument(
        "--model-type",
        type=str.lower,
        choices=tuple(PoetryModel.recurrent_layers),
        default=config.MODEL_TYPE,
        help="选择 PyTorch 内置或手写循环网络",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(model_type=args.model_type)
