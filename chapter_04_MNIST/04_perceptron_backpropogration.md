# 第 4 章：多层感知机（MLP）与反向传播

---

## 🧠 什么是 MLP？

多层感知机（MLP）是由多个线性层（Linear）和非线性激活函数（如 ReLU、Sigmoid）堆叠而成的前馈神经网络。

基本结构如下：

```text
Input → Linear → Activation → Linear → Activation → ... → Output
```

---

## 🔄 什么是反向传播（Backpropagation）？

**反向传播算法（Backpropagation）** 是一种基于链式法则的自动求导方法，用于神经网络中计算各参数的梯度。

### 核心思想：

1. **前向传播**：计算损失值
2. **反向传播**：通过链式法则计算每个参数的偏导数
3. **优化器更新**：使用梯度下降法等算法更新参数

---

## 📐 MLP 的公式推导（两层网络）

设：

* 输入：$x$
* 第1层：$z_1 = W_1x + b_1,\quad a_1 = f(z_1)$
* 第2层：$z_2 = W_2a_1 + b_2,\quad \hat{y} = f(z_2)$
* 损失函数：$L(\hat{y}, y)$

**反向传播时**，我们要计算每层的梯度（偏导数）：

* $\frac{\partial L}{\partial W_2}$
* $\frac{\partial L}{\partial b_2}$
* $\frac{\partial L}{\partial W_1}$
* $\frac{\partial L}{\partial b_1}$

这些可以通过链式法则层层传播。

---

## 🧮 PyTorch 中如何做？

PyTorch 使用自动求导机制（autograd）自动构建计算图并求导。

---

## 📦 示例：训练一个简单的 MLP

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

model = SimpleMLP()
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()   # 清空梯度
    loss.backward()         # 自动反向传播计算梯度
    optimizer.step()        # 使用 optimizer 更新参数

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 🔍 查看梯度

在 `.backward()` 后，每个参数都有 `.grad` 属性：

```python
for name, param in model.named_parameters():
    print(name, param.grad)
```

---

## ✏️ 可视化理解反向传播

### 前向图示意：

```text
Input → Linear1 → ReLU → Linear2 → Sigmoid → Output
```

### 反向传播路径：

```text
Output ← Loss.backward() ← Linear2 ← ReLU ← Linear1 ← Input
```

每一层会缓存中间值（如输入、激活值），用于计算对应梯度。

---

## 🔁 和线性模型的区别？

| 项目     | 线性模型        | MLP 多层网络         |
| ------ | ----------- | ---------------- |
| 非线性能力  | ❌ 无         | ✅ 可拟合复杂函数        |
| 激活函数   | 无           | 有（ReLU, Sigmoid） |
| 参数更新   | 一次反向传播即可    | 多层反向传播           |
| 自动求导支持 | ✅（Autograd） | ✅（Autograd）      |

---

## ❗ 小技巧与注意事项

* `loss.backward()` 前必须调用 `optimizer.zero_grad()` 清除历史梯度
* `optimizer.step()` 必须在 `backward()` 之后调用
* 激活函数如 ReLU 是非线性的，但对梯度计算依然支持
* 可以使用 `torchsummary` 或 `torchinfo` 查看模型结构

---

## ✅ 小结

| 阶段   | 操作               |
| ---- | ---------------- |
| 前向传播 | 输入 → 层层计算 → 输出   |
| 损失计算 | 预测值 vs. 标签 → 损失  |
| 反向传播 | 自动求导 → 得到每个参数的梯度 |
| 参数更新 | 用优化器根据梯度更新权重     |