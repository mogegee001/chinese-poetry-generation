from pathlib import Path


# -------------------- 路径 --------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw" / "ccpc"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

RAW_TRAIN_PATH = RAW_DATA_DIR / "ccpc_train_v1.0.jsonl"
RAW_VALID_PATH = RAW_DATA_DIR / "ccpc_valid_v1.0.jsonl"
RAW_TEST_PATH = RAW_DATA_DIR / "ccpc_test_v1.0.jsonl"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.jsonl"
VALID_PATH = PROCESSED_DATA_DIR / "valid.jsonl"
TEST_PATH = PROCESSED_DATA_DIR / "test.jsonl"
VOCAB_PATH = MODELS_DIR / "vocab.txt"


# -------------------- 数据 --------------------
# 最长样本为七言绝句：
# <bos> + <qiyan> + 28 个汉字 + 3 个 <line> + <eos> = 34
MAX_SEQUENCE_LENGTH = 34
MODEL_INPUT_LENGTH = MAX_SEQUENCE_LENGTH - 1


# -------------------- 模型 --------------------
# 内置实现: "rnn"、"gru"、"lstm"
# 手写实现: "manual_rnn"、"manual_gru"、"manual_lstm"
# 只修改这里即可切换，其余训练、评估和生成代码不需要修改。
MODEL_TYPE = "rnn"
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2


# -------------------- 训练 --------------------
RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3


# -------------------- 生成 --------------------
TEMPERATURE = 0.8
TOP_P = 0.90
# 全诗第一个字的上下文只有 <bos> 和诗体，适当扩大候选集合。
FIRST_TOKEN_TEMPERATURE = 1.0
FIRST_TOKEN_TOP_P = 0.98


def get_checkpoint_path(model_type=None):
    """返回指定循环网络的最佳模型路径。"""
    name = (model_type or MODEL_TYPE).lower()
    return MODELS_DIR / f"best_{name}.pth"
