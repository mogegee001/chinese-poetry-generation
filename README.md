# Chinese Poetry Generation

一个基于 CCPC 数据集的字符级古诗生成学习项目。

基础版当前完成：

1. 用 CCPC 官方训练集、验证集、测试集跑通完整流程。
2. 用字符级 RNN 生成五言或七言绝句。
3. 保留同一套接口，可在 RNN、GRU、LSTM 之间切换。

暂时不使用 `title`、`keywords`、`dynasty`、`author` 等条件，也暂时不加入
注意力、Transformer、格律约束、Web 页面和复杂评价指标。

## 项目结构

```text
chinese-poetry-generation/
├─ data/
│  ├─ raw/ccpc/                 # CCPC 原始 train/valid/test 数据
│  └─ processed/                # process.py 生成的模型输入
├─ logs/                        # TensorBoard 日志
├─ models/
│  ├─ vocab.txt                 # 只根据训练集建立的字符词表
│  └─ best_rnn.pth              # 验证集损失最低的模型
├─ src/
│  ├─ config.py                 # 路径、模型和训练超参数
│  ├─ tokenizer.py              # 字符级编码、解码与词表
│  ├─ process.py                # CCPC 清洗、编码和 padding
│  ├─ dataset.py                # Dataset 与 DataLoader
│  ├─ model.py                  # RNN/GRU/LSTM 语言模型
│  ├─ train.py                  # 训练、验证、早停和模型保存
│  ├─ evaluate.py               # 测试集 loss/PPL/token accuracy
│  └─ predict.py                # 基础五言、七言采样生成
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## 数据表示

七言绝句会被表示为：

```text
<bos> <qiyan> 第一句 <line> 第二句 <line> 第三句 <line> 第四句 <eos>
```

五言绝句使用 `<wuyan>`，并在 `<eos>` 后补 `<pad>`。模型输入是序列的前
33 个 token，监督目标是向右移动一位后的 33 个 token，因此模型在每个时间步
都学习预测下一个汉字或结构符号。

## 运行顺序

在项目根目录执行：

```powershell
pip install -r requirements.txt
python -m src.process
python -m src.dataset
python -m src.train
python -m src.evaluate
python -m src.predict --form 7 --temperature 0.8 --top-k 5
```

生成五言绝句：

```powershell
python -m src.predict --form 5
```

查看训练曲线：

```powershell
tensorboard --logdir logs
```

## 切换 RNN、GRU 和 LSTM

在 `src/config.py` 中修改：

```python
MODEL_TYPE = "rnn"
```

可选值为 `"rnn"`、`"gru"`、`"lstm"`。三个模型使用相同的数据、词表、
训练逻辑和评价指标，模型会分别保存为 `best_rnn.pth`、`best_gru.pth` 和
`best_lstm.pth`，便于后续做公平对比。

## 推荐阅读和修改顺序

1. `config.py`：先看清数据路径、序列长度和超参数。
2. `tokenizer.py`：理解“汉字 -> token id”的过程。
3. `process.py`：理解一首诗怎样变成 input 与 target。
4. `dataset.py`：检查 batch 的形状是否都是 `[batch_size, 33]`。
5. `model.py`：理解 embedding、循环层和 linear 的张量形状。
6. `train.py`：理解前向传播、交叉熵、反向传播、验证和 checkpoint。
7. `evaluate.py`：最后一次在测试集上报告泛化结果。
8. `predict.py`：理解自回归生成以及 temperature、top-k 的作用。

建议先完整训练 RNN 并确认生成流程可用，再只修改 `MODEL_TYPE` 训练 GRU 和
LSTM。不要根据测试集调超参数；超参数选择只看验证集，测试集留给最终比较。

## 数据说明

CCPC 数据集用于学术研究。原始数据和处理后的大文件已在 `.gitignore` 中排除，
发布 GitHub 仓库时请在 README 中保留数据来源与使用说明，不要直接提交模型权重
和数据文件。
