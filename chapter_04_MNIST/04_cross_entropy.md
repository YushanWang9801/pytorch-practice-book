# 第 4 章：分类任务中的 CrossEntropyLoss

`torch.nn.CrossEntropyLoss` 是 PyTorch 中最常用的多类分类损失函数。

---

## 🤔 什么是交叉熵（Cross Entropy）？

交叉熵是用于衡量两个概率分布之间距离的函数。在分类中，它衡量的是**模型预测分布 `ŷ` 与真实标签分布 `y` 之间的差异**。

### 数学定义：

对单个样本：

\[
\text{CrossEntropy}(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i)
\]

- `y` 是 one-hot 编码的真实标签
- `ŷ` 是经过 softmax 的预测概率分布

---

## 🧮 PyTorch 中 CrossEntropyLoss 的实现细节

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, target)
````

### ❗ 注意：PyTorch 内部已包含 `LogSoftmax`

你传入的是：

* `logits`: 模型的原始输出 (未经过 softmax)，shape: **(N, C)**
* `target`: 整数标签，非 one-hot，shape: **(N,)**

### 等价于：

```python
loss = nn.NLLLoss()(F.log_softmax(logits, dim=1), target)
```

---

## 📦 示例：对 3 类分类任务使用 CrossEntropyLoss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设有 3 类，batch size = 2
logits = torch.tensor([[2.0, 0.5, 0.3], [0.1, 0.2, 2.5]])  # shape: (2, 3)
target = torch.tensor([0, 2])  # 每个样本的真实类别（非 one-hot）

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, target)

print("CrossEntropyLoss:", loss.item())
```

---

## 🧠 为什么不自己加 softmax？

因为 `CrossEntropyLoss` 内部在数值上更稳定，采用 **log-softmax + NLLLoss** 的组合可以避免 **浮点数下 softmax 导致的梯度爆炸或消失问题**。

```python
# 正确方式（推荐）
loss = nn.CrossEntropyLoss()(logits, target)

# 错误方式（不要这样！）
prob = F.softmax(logits, dim=1)
loss = nn.NLLLoss()(torch.log(prob), target)
```

---

## 📊 示例：预测概率 & 可视化

```python
import torch.nn.functional as F
probs = F.softmax(logits, dim=1)
print("预测概率分布:\n", probs)
```

---

## 📈 多分类与二分类的区别

| 情况  | 输入              | 标签                 | 使用的 Loss            |
| --- | --------------- | ------------------ | ------------------- |
| 二分类 | 单值 or 2维 logits | float(0\~1) or 0/1 | `BCEWithLogitsLoss` |
| 多分类 | 多类 logits（N, C） | 整数标签（N,）           | `CrossEntropyLoss`  |

---

## ✅ 小结

| 内容        | 描述                                    |
| --------- | ------------------------------------- |
| 输入 logits | 形状为 `(N, C)`，未经 softmax               |
| 标签 target | 整数索引（非 one-hot），形状为 `(N,)`            |
| 内部机制      | 自动执行 `LogSoftmax` + `NLLLoss`         |
| 数值稳定性     | 推荐使用 `CrossEntropyLoss` 而不是手动 softmax |
| 用途        | 多分类任务（包括图像分类、文本分类等）                   |

---

## ➕ 附加提示

* 若你的输出是 one-hot 标签（不推荐），需转换为 class index：

```python
target = torch.argmax(one_hot_target, dim=1)
```

* 如果有类不参与训练（如 padding），使用 `ignore_index`：

```python
nn.CrossEntropyLoss(ignore_index=-1)
```

---