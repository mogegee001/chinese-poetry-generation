import argparse

import torch

from src import config
from src.model import build_model_from_checkpoint, load_checkpoint
from src.tokenizer import CharTokenizer


def sample_next_token(
    logits,
    forbidden_token_ids,
    temperature,
    top_k,
):
    logits = logits.clone()
    logits[list(forbidden_token_ids)] = float("-inf")

    if temperature <= 0:
        return int(logits.argmax().item())

    logits = logits / temperature
    top_k = min(top_k, logits.numel() - len(forbidden_token_ids))
    top_values, top_indexes = torch.topk(logits, k=max(top_k, 1))
    probabilities = torch.softmax(top_values, dim=-1)
    sampled_position = torch.multinomial(probabilities, num_samples=1)
    return int(top_indexes[sampled_position].item())


def generate_poem(
    model,
    tokenizer,
    device,
    form=7,
    temperature=config.TEMPERATURE,
    top_k=config.TOP_K,
):
    if form not in (5, 7):
        raise ValueError("form 只能是 5 或 7")

    form_token = (
        tokenizer.wuyan_token
        if form == 5
        else tokenizer.qiyan_token
    )
    initial_ids = tokenizer.encode_tokens(
        [tokenizer.bos_token, form_token]
    )
    inputs = torch.tensor(
        [initial_ids],
        dtype=torch.long,
        device=device,
    )

    forbidden_token_ids = {
        tokenizer.token2index[token]
        for token in tokenizer.special_tokens
    }
    line_token_id = tokenizer.token2index[tokenizer.line_token]
    generated_lines = []

    model.eval()
    with torch.no_grad():
        logits, state = model(inputs)

        for line_index in range(4):
            current_line = []
            for _ in range(form):
                next_token_id = sample_next_token(
                    logits=logits[0, -1],
                    forbidden_token_ids=forbidden_token_ids,
                    temperature=temperature,
                    top_k=top_k,
                )
                current_line.append(
                    tokenizer.index2token[next_token_id]
                )

                next_input = torch.tensor(
                    [[next_token_id]],
                    dtype=torch.long,
                    device=device,
                )
                logits, state = model(next_input, state)

            generated_lines.append("".join(current_line))
            if line_index < 3:
                line_input = torch.tensor(
                    [[line_token_id]],
                    dtype=torch.long,
                    device=device,
                )
                logits, state = model(line_input, state)

    return (
        f"{generated_lines[0]}，{generated_lines[1]}。\n"
        f"{generated_lines[2]}，{generated_lines[3]}。"
    )


def load_resources(device):
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
            "当前 vocab.txt 与模型训练时使用的词表不一致"
        )

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer, checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="生成一首五言或七言绝句")
    parser.add_argument(
        "--form",
        type=int,
        choices=(5, 7),
        default=7,
        help="每句字数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="0 表示贪心生成；越大随机性越强",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help="只从概率最高的 k 个汉字中采样",
    )
    return parser.parse_args()


def run_predict():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k 必须大于等于 1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, tokenizer, checkpoint = load_resources(device)
    poem = generate_poem(
        model=model,
        tokenizer=tokenizer,
        device=device,
        form=args.form,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"模型: {checkpoint['model_type'].upper()}")
    print(poem)


if __name__ == "__main__":
    run_predict()
