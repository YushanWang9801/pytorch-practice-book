# 第 4 章：感知机（Perceptron）

感知机是最早的神经网络模型之一，用于处理线性可分的二分类问题，是现代深度学习模型的“祖先”。

---

## 🤖 1. 什么是感知机？

感知机是一个线性分类器，输入为特征向量 `x`，输出为类别标签 `y ∈ {0, 1}`。

其核心思想是使用一个**权重向量 `w` 和偏置 `b`**，对输入进行线性变换，并通过激活函数做出决策。

---

## 🧮 2. 数学表达

\[
y = f(w^\top x + b)
\]

其中：
- \( x \in \mathbb{R}^n \)：输入特征
- \( w \in \mathbb{R}^n \)：权重向量
- \( b \in \mathbb{R} \)：偏置
- \( f \)：阶跃函数（step function）

---

### 📏 激活函数（单位阶跃函数）

感知机使用的激活函数是：

\[
f(z) =
\begin{cases}
1 & z \geq 0 \\
0 & z < 0
\end{cases}
\]

---

## 🧠 3. 感知机学习算法

感知机使用的是一种**在线学习算法**，基本思想：

1. 对每个样本预测 `ŷ`
2. 如果预测正确：不更新
3. 如果预测错误：调整参数
   \[
   w := w + \alpha(y - \hat{y}) x \\
   b := b + \alpha(y - \hat{y})
   \]

其中 \( \alpha \) 是学习率。

---

## 📦 4. 用 PyTorch 实现感知机（二分类）

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 模拟一个简单的二维线性可分数据集
X = torch.tensor([
    [2.0, 1.0],
    [1.0, -1.0],
    [-1.0, -2.0],
    [-2.0, 1.0]
])
y = torch.tensor([1, 1, 0, 0]).float().unsqueeze(1)

# 感知机模型（无激活，分类用 threshold 手动处理）
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)  # 二分类用 sigmoid

model = Perceptron(input_dim=2)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练
for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        preds = (y_pred > 0.5).float()
        acc = (preds == y).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}")
```

---

## 🔍 5. 与线性回归的区别

| 项目   | 线性回归       | 感知机          |
| ---- | ---------- | ------------ |
| 目标   | 拟合连续输出     | 分类           |
| 输出   | 任意实数       | 0 or 1       |
| 激活函数 | 无 / Linear | 阶跃或 sigmoid  |
| 损失函数 | MSE        | 0/1 损失 或 BCE |

---

## 📉 6. 感知机的局限性

* ❌ 只能处理线性可分数据
* ❌ 不支持多分类（除非扩展）
* ✅ 可作为神经网络的构件（多个感知机构成 MLP）

---

## 🧩 7. 可视化建议（附加练习）

你可以用 matplotlib 画出分界线：

```python
import matplotlib.pyplot as plt

w, b = model.linear.weight[0].detach(), model.linear.bias.item()
x_line = torch.linspace(-3, 3, 100)
y_line = (-w[0] * x_line - b) / w[1]

plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap='bwr')
plt.plot(x_line, y_line, 'g-')
plt.title("Perceptron Decision Boundary")
plt.show()
```

---

## ✅ 小结

| 概念         | 内容                    |
| ---------- | --------------------- |
| 感知机        | 最早的神经网络模型，用于线性分类      |
| 激活函数       | 阶跃函数（现代用 sigmoid 代替）  |
| 学习规则       | 错了就调整权重               |
| PyTorch 实现 | `Linear + Sigmoid` 即可 |
| 局限         | 无法处理非线性、无法扩展多分类       |