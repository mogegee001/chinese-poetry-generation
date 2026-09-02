import argparse

import torch

from src import config
from src.model import build_model_from_checkpoint, load_checkpoint
from src.tokenizer import CharTokenizer


def sample_next_token(
    logits,
    forbidden_token_ids,
    temperature,
    top_p,
):
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p 必须位于 (0, 1] 区间")

    logits = logits.clone()
    logits[list(forbidden_token_ids)] = float("-inf")

    if temperature <= 0:
        return int(logits.argmax().item())

    logits = logits / temperature
    sorted_logits, sorted_indexes = torch.sort(
        logits,
        descending=True,
    )
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    # 保留第一个使累计概率达到 top_p 的 token。
    remove_mask = cumulative_probabilities > top_p
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False
    sorted_logits[remove_mask] = float("-inf")

    probabilities = torch.softmax(sorted_logits, dim=-1)
    sampled_position = torch.multinomial(probabilities, num_samples=1)
    return int(sorted_indexes[sampled_position].item())


def generate_poem(
    model,
    tokenizer,
    device,
    form=7,
    temperature=config.TEMPERATURE,
    top_p=config.TOP_P,
    first_token_temperature=config.FIRST_TOKEN_TEMPERATURE,
    first_token_top_p=config.FIRST_TOKEN_TOP_P,
):
    if form not in (5, 7):
        raise ValueError("form 只能是 5 或 7")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p 必须位于 (0, 1] 区间")
    if not 0.0 < first_token_top_p <= 1.0:
        raise ValueError(
            "first_token_top_p 必须位于 (0, 1] 区间"
        )

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
            for char_index in range(form):
                is_first_token = (
                    line_index == 0 and char_index == 0
                )
                current_temperature = (
                    first_token_temperature
                    if is_first_token
                    else temperature
                )
                current_top_p = (
                    first_token_top_p
                    if is_first_token
                    else top_p
                )
                next_token_id = sample_next_token(
                    logits=logits[0, -1],
                    forbidden_token_ids=forbidden_token_ids,
                    temperature=current_temperature,
                    top_p=current_top_p,
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
        "--top-p",
        type=float,
        default=config.TOP_P,
        help="从累计概率达到 p 的动态候选集合中采样",
    )
    parser.add_argument(
        "--first-temperature",
        type=float,
        default=config.FIRST_TOKEN_TEMPERATURE,
        help="全诗第一个字使用的 temperature",
    )
    parser.add_argument(
        "--first-top-p",
        type=float,
        default=config.FIRST_TOKEN_TOP_P,
        help="全诗第一个字使用的 top-p",
    )
    return parser.parse_args()


def run_predict():
    args = parse_args()

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
        top_p=args.top_p,
        first_token_temperature=args.first_temperature,
        first_token_top_p=args.first_top_p,
    )
    print(f"模型: {checkpoint['model_type'].upper()}")
    print(poem)


if __name__ == "__main__":
    run_predict()
