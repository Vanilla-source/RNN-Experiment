
import os
import random
import time

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_data
from bigru_model import BiGRUModel


# ==================== 实验参数 ====================
SEED = 42
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
EPOCHS = 10

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    return (
        average_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("使用设备：", device)

    # 与原实验使用相同的数据处理和划分
    train_loader, val_loader, test_loader, vocab = load_data()

    # 保留原数据集划分，只重新创建训练 DataLoader
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = BiGRUModel(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        padding_idx=0
    ).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("BiGRU 可训练参数量：", parameter_count)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_acc = -1.0
    best_epoch = 0
    history = []

    checkpoint_path = os.path.join(
        RESULT_DIR, "bigru_best.pt"
    )

    # ==================== 训练 ====================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()

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

            total_train_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(logits, dim=1)
            correct += (
                predictions == labels
            ).sum().item()
            total += labels.size(0)

        train_loss = (
            total_train_loss / len(train_loader.dataset)
        )
        train_acc = correct / total

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch_time_seconds": epoch_time
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        # 只保留验证集准确率最高的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_acc
            }, checkpoint_path)

    # 保存独立的训练日志
    log_path = os.path.join(
        RESULT_DIR, "bigru_training_log.csv"
    )
    pd.DataFrame(history).to_csv(
        log_path, index=False, encoding="utf-8-sig"
    )

    # ==================== 测试 ====================
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    print("\n========== BiGRU 最终结果 ==========")
    print("参数量：", parameter_count)
    print("最佳 Epoch：", best_epoch)
    print(f"最佳验证集准确率：{best_val_acc:.4f}")
    print(f"测试集 Loss：{test_loss:.4f}")
    print(f"测试集 Accuracy：{test_acc:.4f}")
    print(f"测试集 Precision（正类）：{precision:.4f}")
    print(f"测试集 Recall（正类）：{recall:.4f}")
    print(f"测试集 F1（正类）：{f1:.4f}")
    print("\n分类报告：")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["负面（0）", "正面（1）"],
        zero_division=0
    ))
    print("混淆矩阵：")
    print(cm)

    # 独立保存 BiGRU 的混淆矩阵图片
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("BiGRU Confusion Matrix")
    plt.colorbar()

    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, "bigru_confusion_matrix.png"),
        dpi=200
    )
    plt.close()

    # 保存测试集误分类位置和预测标签
    # 注意：sample_index 是测试集内部序号，不是原 CSV 行号
    errors = np.where(y_true != y_pred)[0]

    error_df = pd.DataFrame({
        "sample_index": errors,
        "true_label": y_true[errors],
        "predicted_label": y_pred[errors]
    })

    error_df.to_csv(
        os.path.join(RESULT_DIR, "bigru_misclassified.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 独立保存汇总，不修改原 model_comparison.csv
    summary = pd.DataFrame([{
        "model": "BiGRU",
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision_positive": precision,
        "test_recall_positive": recall,
        "test_f1_positive": f1,
        "misclassified_count": len(errors)
    }])

    summary.to_csv(
        os.path.join(RESULT_DIR, "bigru_comparison.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nBiGRU 实验完成。")
    print("新增结果保存在 results/ 下的 bigru_* 文件中。")
    print("原 RNN、LSTM、GRU 的结果文件没有被本脚本写入。")


if __name__ == "__main__":
    main()
