import torch
from torch import nn

from src import config
from src.manual_recurrent import ManualGRU, ManualLSTM, ManualRNN


class PoetryModel(nn.Module):
    """Embedding + RNN/GRU/LSTM + Linear"""
    recurrent_layers = {
        "rnn": nn.RNN,
        "gru": nn.GRU,
        "lstm": nn.LSTM,
        "manual_rnn": ManualRNN,
        "manual_gru": ManualGRU,
        "manual_lstm": ManualLSTM,
    }

    def __init__(
        self,
        vocab_size,
        model_type=config.MODEL_TYPE,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        pad_token_id=0,
    ):
        super().__init__()
        model_type = model_type.lower()
        if model_type not in self.recurrent_layers:
            raise ValueError(
                f"model_type 必须是 {tuple(self.recurrent_layers)} 之一"
            )

        self.model_type = model_type
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )
        recurrent_class = self.recurrent_layers[model_type]
        self.recurrent = recurrent_class(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
        )

    def forward(self, inputs, state=None):
        # inputs.shape: [batch_size, sequence_length]
        embeddings = self.embedding(inputs)
        # embeddings.shape: [batch_size, sequence_length, embedding_dim]
        outputs, state = self.recurrent(embeddings, state)
        # outputs.shape: [batch_size, sequence_length, hidden_size]
        logits = self.linear(self.dropout(outputs))
        # logits.shape: [batch_size, sequence_length, vocab_size]
        return logits, state


def build_model_from_checkpoint(checkpoint):
    """根据 checkpoint 中保存的结构参数重建模型。"""
    return PoetryModel(
        vocab_size=checkpoint["vocab_size"],
        model_type=checkpoint["model_type"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        pad_token_id=checkpoint["pad_token_id"],
    )


def load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
