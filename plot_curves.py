
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 结果文件夹
result_dir = Path("results")

# 读取三个模型的训练日志
models = {
    "RNN": "rnn_training_log.csv",
    "LSTM": "lstm_training_log.csv",
    "GRU": "gru_training_log.csv"
}

data = {}

for name, filename in models.items():
    data[name] = pd.read_csv(result_dir / filename)

# 中文字体设置
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# ==============================
# 图1：训练损失曲线
# ==============================
plt.figure(figsize=(9, 5))

for name, df in data.items():
    plt.plot(
        df["epoch"],
        df["train_loss"],
        marker="o",
        linewidth=2,
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.xticks(range(1, 11))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    result_dir / "training_loss_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==============================
# 图2：验证集准确率曲线
# ==============================
plt.figure(figsize=(9, 5))

for name, df in data.items():
    plt.plot(
        df["epoch"],
        df["val_accuracy"] * 100,
        marker="o",
        linewidth=2,
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy Comparison")
plt.xticks(range(1, 11))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    result_dir / "validation_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("训练曲线绘制完成！")
print("已保存到 results 文件夹。")
