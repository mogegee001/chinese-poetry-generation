from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm


class CharTokenizer:
    """面向古诗的字符级分词器。"""

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    line_token = "<line>"
    wuyan_token = "<wuyan>"
    qiyan_token = "<qiyan>"

    special_tokens = [
        pad_token,
        unk_token,
        bos_token,
        eos_token,
        line_token,
        wuyan_token,
        qiyan_token,
    ]

    def __init__(self, vocab_list):
        if len(vocab_list) != len(set(vocab_list)):
            raise ValueError("词表中存在重复 token")
        missing_tokens = [
            token for token in self.special_tokens if token not in vocab_list
        ]
        if missing_tokens:
            raise ValueError(f"词表缺少特殊 token: {missing_tokens}")

        self.vocab_list = list(vocab_list)
        self.vocab_size = len(self.vocab_list)
        self.token2index = {
            token: index for index, token in enumerate(self.vocab_list)
        }
        self.index2token = {
            index: token for index, token in enumerate(self.vocab_list)
        }

    @property
    def pad_token_id(self):
        return self.token2index[self.pad_token]

    @property
    def unk_token_id(self):
        return self.token2index[self.unk_token]

    @staticmethod
    def tokenize(text):
        """中文诗歌采用字符级切分，不依赖外部分词词典。"""
        return list(text)

    def encode(self, text):
        return self.encode_tokens(self.tokenize(text))

    def encode_tokens(self, tokens):
        return [
            self.token2index.get(token, self.unk_token_id)
            for token in tokens
        ]

    def decode(self, indexes, skip_special_tokens=False):
        tokens = [self.index2token[int(index)] for index in indexes]
        if skip_special_tokens:
            tokens = [
                token for token in tokens
                if token not in self.special_tokens
            ]
        return "".join(tokens)

    @classmethod
    def build_vocab(cls, texts: Iterable[str], vocab_path):
        """仅使用训练集构建确定性的字符词表。"""
        vocab_set = set()
        for text in tqdm(texts, desc="构建字符词表"):
            vocab_set.update(cls.tokenize(text))

        vocab_list = cls.special_tokens + sorted(vocab_set)
        vocab_path = Path(vocab_path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text("\n".join(vocab_list), encoding="utf-8")
        print(f"词表大小: {len(vocab_list)}")
        print(f"词表已保存: {vocab_path}")

    @classmethod
    def from_vocab(cls, vocab_path):
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"找不到词表 {vocab_path}，请先运行 python -m src.process"
            )
        vocab_list = vocab_path.read_text(encoding="utf-8").splitlines()
        return cls(vocab_list)
