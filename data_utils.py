import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ==================== 1. 固定随机种子 ====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==================== 2. 文本预处理 ====================

def tokenize(text):
    """采用字级分词：将中文文本拆分为单个字符。"""
    text = str(text).strip()
    return list(text)


# ==================== 3. 构建词表 ====================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocab(texts, max_vocab_size=20000):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    # 0 留给填充，1 留给未知字符
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for token, _ in counter.most_common(max_vocab_size - 2):
        vocab[token] = len(vocab)

    return vocab


# ==================== 4. 文本转数字 ====================

def encode_text(text, vocab, max_len=200):
    tokens = tokenize(text)
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

    # 空文本至少保留一个位置
    if not token_ids:
        token_ids = [vocab[UNK_TOKEN]]

    # 截断过长文本
    token_ids = token_ids[:max_len]
    length = len(token_ids)

    # 补齐到统一长度
    token_ids += [vocab[PAD_TOKEN]] * (max_len - length)

    return token_ids, length


# ==================== 5. 自定义数据集 ====================

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids, length = encode_text(
            self.texts[idx], self.vocab, self.max_len
        )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(int(self.labels[idx]), dtype=torch.long)
        )


# ==================== 6. 加载并划分数据 ====================

def load_data(
    csv_path="data/ChnSentiCorp_htl_all.csv",
    batch_size=64,
    max_vocab_size=20000,
    max_len=200,
    seed=42
):
    set_seed(seed)

    df = pd.read_csv(csv_path)

    # 检查必要字段
    required_columns = {"label", "review"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV 必须包含列 {required_columns}，实际列名为 {df.columns.tolist()}"
        )

    df = df[["label", "review"]].dropna().copy()
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"] != ""].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label 只能为 0（负面）或 1（正面）")

    texts = df["review"].tolist()
    labels = df["label"].tolist()

    # 先划出 10% 测试集
    train_val_texts, test_texts, train_val_labels, test_labels = (
        train_test_split(
            texts,
            labels,
            test_size=0.1,
            random_state=seed,
            stratify=labels
        )
    )

    # 再从剩余 90% 中划出 1/9，得到总数据量约 10% 的验证集
    train_texts, val_texts, train_labels, val_labels = (
        train_test_split(
            train_val_texts,
            train_val_labels,
            test_size=1 / 9,
            random_state=seed,
            stratify=train_val_labels
        )
    )

    # 只使用训练集构建词表，防止验证集和测试集信息泄漏
    vocab = build_vocab(train_texts, max_vocab_size)

    train_dataset = SentimentDataset(
        train_texts, train_labels, vocab, max_len
    )
    val_dataset = SentimentDataset(
        val_texts, val_labels, vocab, max_len
    )
    test_dataset = SentimentDataset(
        test_texts, test_labels, vocab, max_len
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("数据集加载成功！")
    print(f"训练集：{len(train_dataset)} 条")
    print(f"验证集：{len(val_dataset)} 条")
    print(f"测试集：{len(test_dataset)} 条")
    print(f"词表大小：{len(vocab)}")
    print(f"最大文本长度：{max_len}")
    print(f"批次大小：{batch_size}")

    return train_loader, val_loader, test_loader, vocab
