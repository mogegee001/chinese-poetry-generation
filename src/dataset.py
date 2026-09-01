import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src import config


class PoetryDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到处理后的数据 {path}，"
                "请先运行 python -m src.process"
            )

        self.data = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        input_tensor = torch.tensor(sample["input"], dtype=torch.long)
        target_tensor = torch.tensor(sample["target"], dtype=torch.long)
        return input_tensor, target_tensor


def get_dataloader(split="train", batch_size=None, shuffle=None):
    split_paths = {
        "train": config.TRAIN_PATH,
        "valid": config.VALID_PATH,
        "test": config.TEST_PATH,
    }
    if split not in split_paths:
        raise ValueError(
            f"split 必须是 {tuple(split_paths)} 之一，实际为 {split!r}"
        )

    if shuffle is None:
        shuffle = split == "train"

    dataset = PoetryDataset(split_paths[split])
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


if __name__ == "__main__":
    for split in ("train", "valid", "test"):
        dataloader = get_dataloader(split)
        inputs, targets = next(iter(dataloader))
        print(
            f"{split}: batches={len(dataloader)}, "
            f"inputs={tuple(inputs.shape)}, "
            f"targets={tuple(targets.shape)}"
        )
