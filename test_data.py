from data_utils import load_data

# 加载数据集
train_loader, val_loader, test_loader, vocab = load_data()

# 获取训练集的第一个批次
input_ids, lengths, labels = next(iter(train_loader))


# 输出检查结果
print("\n========== 数据加载测试结果 ==========")

print("数据集加载成功！")

print(f"训练集样本数：{len(train_loader.dataset)}")
print(f"验证集样本数：{len(val_loader.dataset)}")
print(f"测试集样本数：{len(test_loader.dataset)}")

print(f"词表大小：{len(vocab)}")

print("\n========== 首个批次检查 ==========")

print("文本张量形状：", input_ids.shape)
print("长度张量形状：", lengths.shape)
print("标签张量形状：", labels.shape)

print("前 5 个标签：", labels[:5].tolist())
print("当前批次标签取值：", sorted(set(labels.tolist())))

print("\n========== 测试完成 ==========")
