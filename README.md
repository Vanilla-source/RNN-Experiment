# 深度学习实验：ChnSentiCorp 酒店评论情感分类（RNN / LSTM / GRU / BiGRU）

第二组课程实验配套代码。在统一预处理和超参数下，训练并对比三种循环神经网络
（RNN、LSTM、GRU），并以 BiGRU 作为扩展模型，完成酒店评论的情感二分类。

## 1. 环境要求

- Python 3.12+（实测 3.13）
- 依赖见 `requirements.txt`，安装命令：

```bash
pip install -r requirements.txt
```

## 2. 数据放置

数据集已随仓库附带，无需单独下载：

```
data/ChnSentiCorp_htl_all.csv   # ChnSentiCorp 酒店评论数据，共 7765 条
```

`data_utils.py` 默认读取该路径（`data/ChnSentiCorp_htl_all.csv`），列名须包含
`review`（评论文本）和 `label`（0 = 负面，1 = 正面）。如需更换数据，替换该 CSV
并保持列名一致即可。

数据划分（seed=42，分层抽样）：训练集 6211 条（80%）/ 验证集 777 条（10%）/
测试集 777 条（10%）。

## 3. 运行方式

```bash
# 1) 数据管道自检（可选）
python test_data.py

# 2) 三个基础模型训练 + 测试集评估（输出混淆矩阵、误分类样本、训练日志）
python train.py

# 3) BiGRU 扩展实验
python train_bigru.py

# 4) 绘制三模型训练损失 / 验证准确率对比曲线（可选）
python plot_curves.py

# 5) 查看误分类案例（可选）
python view_errors.py
```

完整训练一轮约 30 分钟（CPU）。已训练好的模型权重保存在 `results/` 下
（`*_best.pt`），不重训也可直接复现测试集结果。

## 4. 输出说明

所有结果写入 `results/` 目录：

| 文件 | 说明 |
|---|---|
| `{model}_best.pt` | 验证集准确率最高的模型权重 |
| `{model}_training_log.csv` | 逐轮训练损失 / 训练准确率 / 验证损失 / 验证准确率 / 耗时 |
| `{model}_confusion_matrix.png` | 测试集混淆矩阵 |
| `{model}_misclassified.csv` | 测试集误分类样本明细 |
| `model_comparison.csv` | 三基础模型指标汇总（参数量、最佳轮次、测试指标、每轮耗时） |
| `bigru_comparison.csv` | BiGRU 指标汇总 |

## 5. 主要结果（测试集）

| 模型 | 参数量 | 最佳验证 Acc | 测试 Accuracy | 测试 F1 | 平均每轮耗时 |
|---|---|---|---|---|---|
| RNN | 483,074 | 77.99% | 72.97% | 80.37% | 13.1 s |
| LSTM | 582,146 | 82.24% | 80.69% | 85.85% | 36.2 s |
| GRU | 549,122 | 88.93% | 86.87% | 90.50% | 25.4 s |
| BiGRU（扩展） | 648,450 | 86.74% | 86.10% | 90.07% | 108.7 s |

结论：GRU 在参数量小于 LSTM 的情况下取得最优测试指标，训练速度也更快；
BiGRU 通过双向编码进一步提升对正面类的召回（Recall 91.93%）。
