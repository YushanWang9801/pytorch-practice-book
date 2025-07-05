# 第 2 章：Tensor 与自动求导（Autograd）基础

在本章中，你将学习 PyTorch 的核心概念：`Tensor` 和 `autograd`，并通过代码掌握以下知识点：

- Tensor 的基本操作
- Tensor 与 NumPy 的转换
- Tensor 的维度变换、切片、索引
- 自动求导机制（autograd）
- 使用 `requires_grad`、`backward()`、`.grad` 查看梯度

---

## 🎯 什么是 Tensor？

Tensor 是 PyTorch 中的基本数据结构，和 NumPy 的 `ndarray` 类似，但更强大。  
主要特点：

- 支持 GPU 加速
- 可以自动求导
- 与 NumPy 无缝转换

---

## 🛠️ Tensor 的创建

```python
import torch

# 创建全 0 Tensor
x = torch.zeros((2, 3))
print(x)

# 创建随机 Tensor
x = torch.rand((2, 3))
print(x)

# 直接创建
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x)
```

### 指定 dtype 和 device

```python
x = torch.ones((2, 2), dtype=torch.float32, device="cpu")  # 或 device="cuda" 如果支持
```

---

## 🔁 NumPy 与 Tensor 互转

```python
import numpy as np

a = np.array([1, 2, 3])
t = torch.from_numpy(a)

n = t.numpy()
```

注意：这两者共享内存，修改其中一个，另一个也会变！

---

## 🔍 Tensor 的基本操作

```python
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

# 加法
z = x + y
z = torch.add(x, y)

# 乘法（逐元素）
z = x * y

# 矩阵乘法
z = torch.matmul(x, y)
```

---

## 🔄 Tensor 的维度操作

```python
x = torch.rand(2, 3)

# 查看形状
print(x.shape)

# 改变形状
x_reshaped = x.view(3, 2)

# 添加维度
x_unsqueeze = x.unsqueeze(0)  # 在第 0 维添加一维

# 去掉维度
x_squeezed = x_unsqueeze.squeeze()
```

---

## 📈 自动求导 Autograd

PyTorch 能根据前向传播自动构建计算图，并进行反向传播。

### 基本示例

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x^2 + 3x + 1

y.backward()  # 计算 dy/dx

print(x.grad)  # 输出梯度：dy/dx = 2x + 3 = 7
```

---

## ⚠️ `requires_grad` 的作用

只有 `requires_grad=True` 的 Tensor 才会参与计算图的构建。

```python
x = torch.tensor([2.0], requires_grad=True)
```

要禁止 autograd 的情况（例如评估模型时）：

```python
with torch.no_grad():
    y = model(x)
```

或者：

```python
x = x.detach()
```

---

## 🧮 更复杂的自动求导（链式）

```python
x = torch.tensor(1.0, requires_grad=True)
y = x * 2
z = y ** 2 + 3

z.backward()
print(x.grad)  # dz/dx = d(y^2 + 3)/dx = 4x = 4
```

---

## 🎓 小练习（建议）

1. 创建一个三维 Tensor，计算其某个维度的均值
2. 构造一个多步的计算图，并打印每一步的 `.grad_fn`
3. 尝试 `.backward()` 多次，观察行为

---

## ✅ 小结

* Tensor 是 PyTorch 的核心数据结构，支持多维数组和 GPU 加速
* `requires_grad=True` 可以让 Tensor 参与自动求导
* `backward()` 会根据计算图自动反向传播梯度
* `.grad` 可以查看梯度值