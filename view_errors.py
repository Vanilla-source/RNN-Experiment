
import os
import pandas as pd

RESULT_DIR = "results"

models = ["rnn", "lstm", "gru"]

for model in models:
    file_path = os.path.join(
        RESULT_DIR, f"{model}_misclassified.csv"
    )

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"{model.upper()}：误分类案例")
    print(f"误分类总数：{len(df)}")
    print("=" * 70)

    # 尽量各选一条不同方向的错误
    selected = []

    for error_type in ["负面误判为正面", "正面误判为负面"]:
        group = df[df["error_type"] == error_type]
        if not group.empty:
            selected.append(group.iloc[0])

    # 如果某个方向没有案例，就用其他错误补足
    if len(selected) < 2:
        for _, row in df.iterrows():
            if len(selected) >= 2:
                break
            if not any(
                row["review"] == item["review"]
                and row["true_label"] == item["true_label"]
                and row["predicted_label"] == item["predicted_label"]
                for item in selected
            ):
                selected.append(row)

    for index, row in enumerate(selected[:2], start=1):
        print(f"\n案例 {index}")
        print("评论原文：", row["review"])
        print("真实标签：", row["true_label"])
        print("预测标签：", row["predicted_label"])
        print("错误类型：", row["error_type"])
