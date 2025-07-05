# 第 3 章：激活函数与损失函数（Activation & Loss）

在构建神经网络时，**激活函数** 决定了模型的非线性能力，**损失函数** 则定义了模型的优化目标。

---

## 🔋 1. 激活函数（Activation Functions）

### 为什么需要激活函数？

没有激活函数的神经网络本质上是一个线性变换：
\[
y = W_2(W_1x + b_1) + b_2
\]
加再多层都是线性的组合。而激活函数引入**非线性变换**，让网络具备拟合复杂函数的能力。

---

### ✅ 常见激活函数及其性质

| 函数 | 公式 | 范围 | 导数是否连续 | 是否饱和 | 应用 |
|------|------|------|---------------|-----------|------|
| ReLU | \( \max(0, x) \) | \([0, \infty)\) | 不连续 | 否 | 默认推荐 |
| Leaky ReLU | \( \max(0.01x, x) \) | \((-\infty, \infty)\) | 连续 | 否 | 避免 ReLU 死亡 |
| Sigmoid | \( \frac{1}{1 + e^{-x}} \) | \((0,1)\) | 连续 | 是 | 输出为概率 |
| Tanh | \( \tanh(x) \) | \((-1, 1)\) | 连续 | 是 | 零中心 |
| GELU | 近似 \( x \Phi(x) \) | 类似 ReLU | 连续 | 否 | Transformer 默认 |

---

### 📦 PyTorch 中的激活函数调用方式

#### 1. 作为模块（用于 `nn.Sequential`）
```python
import torch.nn as nn

nn.ReLU()
nn.Tanh()
nn.Sigmoid()
nn.GELU()
```

#### 2. 作为函数（用于函数式 API）

```python
import torch.nn.functional as F

F.relu(x)
F.sigmoid(x)
F.tanh(x)
F.gelu(x)
```

---

### 🧮 各激活函数的导数（自动求导）

```python
import torch
x = torch.tensor([1.0], requires_grad=True)
y = torch.relu(x)
y.backward()
print(x.grad)  # 1.0
```

你无需手动写导数，PyTorch 会自动根据计算图反向传播。

---

## 📉 2. 损失函数（Loss Functions）

### 损失函数的作用？

定义了模型输出 `ŷ` 与真实标签 `y` 之间的“距离”。优化器会尝试最小化这个距离。

---

### ✅ 常见损失函数一览

#### 1. MSE Loss（均方误差）

用于回归任务：

$$
\text{MSE}(y, \hat{y}) = \frac{1}{N} \sum (y_i - \hat{y}_i)^2
$$

```python
loss = nn.MSELoss()
```

#### 2. L1 Loss（绝对值误差）

$$
\text{L1}(y, \hat{y}) = \frac{1}{N} \sum |y_i - \hat{y}_i|
$$

```python
loss = nn.L1Loss()
```

#### 3. CrossEntropy Loss（交叉熵）

用于分类任务，等价于：

```python
nn.CrossEntropyLoss() == nn.LogSoftmax + nn.NLLLoss
```

输入要求：

* `input`: shape = (N, C) → logits（未经过 softmax 的输出）
* `target`: shape = (N,) → 类别索引（如 `0`, `1`, `2`）

```python
loss = nn.CrossEntropyLoss()
output = model(x)      # shape = (batch_size, num_classes)
loss_val = loss(output, target)
```

#### 4. BCE Loss（二分类）

用于输出为概率的二分类任务（输入需经过 sigmoid）

```python
loss = nn.BCELoss()
F.sigmoid(output) → loss(...)
```

或更推荐使用：

```python
loss = nn.BCEWithLogitsLoss()  # 内部已包含 sigmoid
loss(output, target)
```

---

### 📊 示例：计算并反向传播损失

```python
import torch
import torch.nn as nn

y_pred = torch.tensor([[0.3], [0.7]], requires_grad=True)
y_true = torch.tensor([[0.0], [1.0]])

loss_fn = nn.BCELoss()
loss = loss_fn(y_pred, y_true)
loss.backward()

print(loss.item())
```

---

## 🤔 总结：如何选择激活函数和损失函数？

| 任务类型 | 输出层激活                           | 损失函数                 |
| ---- | ------------------------------- | -------------------- |
| 回归   | 无 / Linear                      | `MSELoss` / `L1Loss` |
| 二分类  | `Sigmoid` 或 `BCEWithLogitsLoss` | `BCEWithLogitsLoss`  |
| 多分类  | `Softmax`（由 CrossEntropy 内部处理）  | `CrossEntropyLoss`   |

---

## 🛠 训练示意图（以二分类为例）

```python
model = nn.Sequential(
    nn.Linear(10, 1),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## ✅ 小结

| 分类   | 函数                                           | 说明             |
| ---- | -------------------------------------------- | -------------- |
| 激活函数 | `ReLU`, `Sigmoid`, `Tanh`, `GELU`            | 控制非线性能力        |
| 损失函数 | `MSE`, `L1`, `CrossEntropy`, `BCEWithLogits` | 衡量模型好坏         |
| 自动求导 | `.backward()`                                | 自动链式求导，无需显式写梯度 |