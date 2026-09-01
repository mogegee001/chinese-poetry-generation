import json
from pathlib import Path

from tqdm import tqdm

from src import config
from src.tokenizer import CharTokenizer


def iter_jsonl(path):
    """逐行读取 JSONL，避免一次把原始数据全部载入内存。"""
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON"
                ) from error


def parse_poem(record):
    """
    将 CCPC 的 content 字段解析成四句五言或七言绝句。

    返回:
        (lines, form_token)，不符合基础版要求的数据返回 None。
    """
    content = str(record.get("content", "")).strip()
    lines = [line.strip() for line in content.split("|")]

    if len(lines) != 4 or any(not line for line in lines):
        return None

    line_lengths = {len(line) for line in lines}
    if line_lengths == {5}:
        return lines, CharTokenizer.wuyan_token
    if line_lengths == {7}:
        return lines, CharTokenizer.qiyan_token
    return None


def iter_training_texts():
    for record in iter_jsonl(config.RAW_TRAIN_PATH):
        parsed = parse_poem(record)
        if parsed is not None:
            lines, _ = parsed
            yield "".join(lines)


def encode_poem(lines, form_token, tokenizer):
    tokens = [tokenizer.bos_token, form_token]
    for index, line in enumerate(lines):
        tokens.extend(tokenizer.tokenize(line))
        if index < len(lines) - 1:
            tokens.append(tokenizer.line_token)
    tokens.append(tokenizer.eos_token)

    token_ids = tokenizer.encode_tokens(tokens)
    if len(token_ids) > config.MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"样本长度 {len(token_ids)} 超过上限 "
            f"{config.MAX_SEQUENCE_LENGTH}"
        )

    token_ids += [tokenizer.pad_token_id] * (
        config.MAX_SEQUENCE_LENGTH - len(token_ids)
    )
    return {
        "input": token_ids[:-1],
        "target": token_ids[1:],
    }


def process_split(raw_path, output_path, tokenizer):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_count = 0
    skipped_count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in tqdm(
            iter_jsonl(raw_path),
            desc=f"处理 {raw_path.name}",
        ):
            parsed = parse_poem(record)
            if parsed is None:
                skipped_count += 1
                continue

            lines, form_token = parsed
            sample = encode_poem(lines, form_token, tokenizer)
            output_file.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )
            kept_count += 1

    print(
        f"{output_path.name}: 保留 {kept_count} 首，"
        f"跳过 {skipped_count} 首"
    )


def process():
    print("开始处理 CCPC 数据")
    required_paths = [
        config.RAW_TRAIN_PATH,
        config.RAW_VALID_PATH,
        config.RAW_TEST_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"缺少原始数据文件:\n{missing_text}")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 只从训练集建立词表，防止验证集和测试集信息泄漏。
    CharTokenizer.build_vocab(
        iter_training_texts(),
        config.VOCAB_PATH,
    )
    tokenizer = CharTokenizer.from_vocab(config.VOCAB_PATH)

    split_paths = [
        (config.RAW_TRAIN_PATH, config.TRAIN_PATH),
        (config.RAW_VALID_PATH, config.VALID_PATH),
        (config.RAW_TEST_PATH, config.TEST_PATH),
    ]
    for raw_path, output_path in split_paths:
        process_split(raw_path, output_path, tokenizer)

    print("数据处理完成")


if __name__ == "__main__":
    process()
