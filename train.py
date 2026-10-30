
import os
import time
import random
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

from data_utils import load_data, set_seed
from rnn_model import RNNModel
from lstm_model import LSTMModel
from gru_model import GRUModel


# ==================== 1. 实验参数 ====================

SEED = 42
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_DIM = 128
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
EPOCHS = 10

DATA_PATH = os.path.join(
    "data", "ChnSentiCorp_htl_all.csv"
)
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)


# ==================== 2. 固定随机种子 ====================

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== 3. 计算模型参数量 ====================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ==================== 4. 验证或测试模型 ====================

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for input_ids, lengths, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(data_loader.dataset)

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    return (
        average_loss,
        accuracy,
        all_labels,
        all_predictions
    )


# ==================== 5. 绘制混淆矩阵 ====================

def save_confusion_matrix(y_true, y_pred, model_name):
    matrix = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        title=f"{model_name} Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"]
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black"
            )

    fig.tight_layout()

    save_path = os.path.join(
        RESULT_DIR, f"{model_name}_confusion_matrix.png"
    )
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return matrix


# ==================== 6. 训练单个模型 ====================

def train_one_model(
    model_name,
    model_class,
    vocab_size,
    train_dataset,
    val_loader,
    test_loader,
    device
):
    print("\n" + "=" * 60)
    print(f"开始训练：{model_name.upper()}")
    print("=" * 60)

    # 每个模型使用相同的随机种子
    fix_seed(SEED)

    # 为每个模型重新创建训练加载器，
    # 使训练数据的打乱顺序可复现
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    # 创建模型
    model = model_class(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        padding_idx=0
    ).to(device)

    parameter_count = count_parameters(model)
    print(f"可训练参数量：{parameter_count:,}")
    print(f"训练设备：{device}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = None

    epoch_records = []

    # ---------- 训练循环 ----------
    for epoch in range(1, EPOCHS + 1):
        start_time = time.perf_counter()

        model.train()
        total_train_loss = 0.0
        correct = 0
        total = 0

        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_train_loss += loss.item() * batch_size

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

        train_loss = total_train_loss / total
        train_accuracy = correct / total

        # 每个 Epoch 只在验证集上评估
        val_loss, val_accuracy, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.perf_counter() - start_time

        record = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_time_seconds": epoch_time
        }
        epoch_records.append(record)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"耗时: {epoch_time:.2f}s"
        )

        # 保存验证集准确率最高的模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # 保存每轮训练记录
    epoch_log_path = os.path.join(
        RESULT_DIR, f"{model_name}_training_log.csv"
    )
    pd.DataFrame(epoch_records).to_csv(
        epoch_log_path, index=False, encoding="utf-8-sig"
    )

    # 保存最佳模型
    checkpoint_path = os.path.join(
        RESULT_DIR, f"{model_name}_best.pt"
    )
    torch.save(best_state, checkpoint_path)

    print(f"\n最佳 Epoch：{best_epoch}")
    print(f"最佳验证集准确率：{best_val_accuracy:.4f}")
    print(f"模型已保存：{checkpoint_path}")

    # ---------- 最终测试 ----------
    # 加载验证集准确率最高的模型，再评估测试集
    model.load_state_dict(best_state)

    _, _, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    matrix = save_confusion_matrix(
        y_true, y_pred, model_name
    )

    print(f"\n{model_name.upper()} 测试集结果")
    print(f"Accuracy ：{accuracy:.4f}")
    print(f"Precision：{precision:.4f}")
    print(f"Recall   ：{recall:.4f}")
    print(f"F1       ：{f1:.4f}")
    print("\n分类报告：")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["负面（0）", "正面（1）"],
            zero_division=0
        )
    )
    print("混淆矩阵：")
    print(matrix)

    # 保存误分类评论，供实验报告分析
    test_texts = test_loader.dataset.texts

    misclassified = []
    for index, (true_label, pred_label) in enumerate(
        zip(y_true, y_pred)
    ):
        if true_label != pred_label:
            misclassified.append({
                "model": model_name,
                "review": test_texts[index],
                "true_label": true_label,
                "predicted_label": pred_label,
                "error_type": (
                    "负面误判为正面"
                    if true_label == 0
                    else "正面误判为负面"
                )
            })

    error_path = os.path.join(
        RESULT_DIR, f"{model_name}_misclassified.csv"
    )
    pd.DataFrame(
        misclassified,
        columns=[
            "model", "review", "true_label",
            "predicted_label", "error_type"
        ]
    ).to_csv(
        error_path, index=False, encoding="utf-8-sig"
    )

    print(f"误分类评论数量：{len(misclassified)}")
    print(f"误分类记录已保存：{error_path}")

    # 汇总指标
    return {
        "model": model_name,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "mean_epoch_time_seconds": np.mean([
            item["epoch_time_seconds"]
            for item in epoch_records
        ]),
        "total_training_time_seconds": sum([
            item["epoch_time_seconds"]
            for item in epoch_records
        ]),
        "misclassified_count": len(misclassified)
    }


# ==================== 7. 主程序 ====================

def main():
    fix_seed(SEED)

    # 本实验使用 CPU 或自动检测到的 CUDA 设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("设备：", device)
    print("正在加载数据集……")

    train_loader, val_loader, test_loader, vocab = load_data(
        csv_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_len=MAX_LEN,
        seed=SEED
    )

    # 取出同一份训练数据集，供三个模型公平比较
    train_dataset = train_loader.dataset

    models = [
        ("rnn", RNNModel),
        ("lstm", LSTMModel),
        ("gru", GRUModel)
    ]

    all_results = []

    for model_name, model_class in models:
        result = train_one_model(
            model_name=model_name,
            model_class=model_class,
            vocab_size=len(vocab),
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device
        )
        all_results.append(result)

        # 每完成一个模型，立即保存当前汇总结果
        summary_path = os.path.join(
            RESULT_DIR, "model_comparison.csv"
        )
        pd.DataFrame(all_results).to_csv(
            summary_path, index=False, encoding="utf-8-sig"
        )

    print("\n" + "=" * 60)
    print("三个模型训练与测试全部完成")
    print("=" * 60)

    comparison = pd.DataFrame(all_results)
    print(
        comparison[
            [
                "model",
                "parameter_count",
                "best_epoch",
                "best_val_accuracy",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1"
            ]
        ].to_string(index=False)
    )

    print(f"\n所有结果保存在：{os.path.abspath(RESULT_DIR)}")


if __name__ == "__main__":
    main()
