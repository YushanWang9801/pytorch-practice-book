# 第 3 章：线性回归与梯度下降（Gradient Descent）

在这一章中，我们将从零开始实现一个简单的一元线性回归模型，学习以下内容：

- 什么是回归（Regression）
- 什么是梯度下降（Gradient Descent）
- 用纯 Python/Numpy 实现一个回归模型
- 用 PyTorch 重写该模型，使用 `SGD` 优化器
- 数据文件格式与运行说明

---

## 📈 什么是回归？

回归是一种监督学习任务，用于预测一个**连续值**。

最常见的是**一元线性回归**，其数学表达是：

\[
y = wx + b
\]

其中：

- `x` 是输入（自变量）
- `y` 是输出（因变量）
- `w` 是斜率（权重）
- `b` 是偏置项（截距）

我们的目标是从一堆训练数据中学习出最佳的 `w` 和 `b`，使得预测值 `y` 尽可能接近真实值。

---

## 📉 什么是梯度下降（Gradient Descent）？

梯度下降是一种优化算法，用来最小化损失函数（如均方误差 MSE）。

目标函数（MSE）定义为：

\[
\text{MSE} = \frac{1}{N} \sum\_{i=1}^N (y_i - (w x_i + b))^2
\]

我们不断更新 `w` 和 `b`：

\[
w := w - \alpha \cdot \frac{\partial L}{\partial w}  
b := b - \alpha \cdot \frac{\partial L}{\partial b}
\]

其中：

- `α` 是学习率（learning rate）
- `∂L/∂w` 是损失函数对参数的梯度

---

## 🧮 原始 Python 实现

我们手动实现了梯度下降算法，文件为：

```python
gradient_descent_numpy.py
```

该文件包含：

- 手写 `compute_error_for_line_given_points()` 计算 MSE
- 手写 `step_gradient()` 计算每一步的梯度
- 手写参数更新过程
- 加载 `data.csv` 文件并打印初始与最终误差

运行：

```bash
python gradient_descent_numpy.py
```

---

## 🚀 使用 PyTorch 实现线性回归

我们重写了一个 PyTorch 版本，文件为：

```python
linear_regression_pytorch.py
```

特性：

- 使用 `torch.tensor()` 存储输入与输出
- `requires_grad=True` 启用自动求导
- 使用 `torch.optim.SGD` 优化器更新参数
- 支持 GPU（自动检测）
- 可视化训练结果（matplotlib）

运行方式：

```bash
python linear_regression_pytorch.py
```

---

## 🧰 关键模块解释

### ✅ 自动求导

PyTorch 自动为你构建计算图，执行 `.backward()` 就能自动计算梯度：

```python
loss = torch.mean((y - y_pred) ** 2)
loss.backward()
```

### ✅ 优化器（SGD）

使用 `torch.optim.SGD` 替代手动梯度更新：

```python
optimizer = torch.optim.SGD([w, b], lr=0.0001)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 📂 数据文件格式说明：`data.csv`

必须是一个两列的纯文本 CSV 文件，每行格式如下：

```csv
x1, y1
x2, y2
x3, y3
...
```

### 示例：

```csv
1.0, 2.1
2.0, 4.2
3.0, 6.3
4.0, 8.1
```

该文件在两个脚本中都会被读取：

```python
points = np.genfromtxt("data.csv", delimiter=",")
```

或：

```python
open("data.csv")
```

---

## ✅ 小结

| 内容     | 实现方式                     |
| -------- | ---------------------------- |
| 回归模型 | `y = wx + b`                 |
| 优化目标 | 最小化 MSE                   |
| 梯度下降 | 手动实现 or PyTorch 自动求导 |
| 优化器   | 手写更新 / PyTorch 的 `SGD`  |
| 数据来源 | CSV 文件                     |
| 运行方式 | 直接执行 Python 脚本         |
