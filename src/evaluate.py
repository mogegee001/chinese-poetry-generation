import math

import torch
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.model import build_model_from_checkpoint, load_checkpoint
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
    }


def run_evaluate():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = config.get_checkpoint_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型 {checkpoint_path}，"
            "请先运行 python -m src.train"
        )

    tokenizer = CharTokenizer.from_vocab(config.VOCAB_PATH)
    checkpoint = load_checkpoint(checkpoint_path, device)
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

    metrics = evaluate(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"模型: {checkpoint['model_type'].upper()}")
    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"test_loss: {metrics['loss']:.4f}")
    print(f"perplexity: {metrics['perplexity']:.4f}")
    print(f"token_accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    run_evaluate()
