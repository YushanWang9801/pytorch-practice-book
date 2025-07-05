# 附录：Tensor 操作大全（索引、变形、维度、广播等）

本章补充 PyTorch 中常见的 Tensor 操作，包括维度变换、索引、转置与广播等，适用于处理图像、文本、序列等不同形状的数据。

---

## 🔍 索引与切片

PyTorch Tensor 的索引和 NumPy 类似。

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x[0])      # 第一行
print(x[:, 1])   # 第二列
print(x[1, 2])   # 第二行第三列
print(x[-1])     # 最后一行
```

---

## 🔄 View / Reshape

`view()` 和 `reshape()` 都可以改变 Tensor 的形状。

```python
x = torch.arange(12)
x1 = x.view(3, 4)       # 原地重构，要求内存连续
x2 = x.reshape(2, 6)    # 更灵活，支持非连续内存

print(x1)
print(x2)
```

---

## 🪞 Squeeze / Unsqueeze

`squeeze()` 去除维度为 1 的维度；`unsqueeze()` 增加维度。

```python
x = torch.rand(1, 3, 1)

x1 = x.squeeze()           # (1,3,1) → (3)
x2 = x.unsqueeze(0)        # (1,3,1) → (1,1,3,1)

print(x.shape, x1.shape, x2.shape)
```

常用于数据准备、批处理、卷积输入等。

---

## 🔁 Transpose / .t() / Permute

### 1. `.t()` 只适用于 2D Tensor

```python
x = torch.tensor([[1, 2], [3, 4]])
print(x.t())  # 转置
```

### 2. `transpose(dim0, dim1)`

```python
x = torch.rand(2, 3, 4)
x_t = x.transpose(1, 2)  # 交换第1和第2维
```

### 3. `permute(*dims)` 可任意重排所有维度

```python
x = torch.rand(2, 3, 4)
x_p = x.permute(2, 0, 1)  # (4, 2, 3)
```

---

## 🔁 Expand / Repeat

用于广播扩展张量维度

### `expand()`：共享内存（仅支持扩展为 1 的维度）

```python
x = torch.tensor([[1], [2], [3]])  # shape = (3, 1)
x_exp = x.expand(3, 4)             # shape = (3, 4)
```

### `repeat()`：真实复制数据

```python
x = torch.tensor([1, 2, 3])
x_rep = x.repeat(2, 3)  # shape = (2, 9)
```

| 区别     | expand | repeat |
| ------ | ------ | ------ |
| 内存共享   | ✅      | ❌      |
| 是否复制数据 | 否      | 是      |
| 是否快    | ✅      | 慢      |

---

## 🎲 常用随机数操作

```python
# 均匀分布 [0,1)
torch.rand(2, 3)

# 标准正态分布 N(0,1)
torch.randn(2, 3)

# 整数均匀分布
torch.randint(low=0, high=10, size=(2, 3))

# 均匀分布 [a,b]
torch.empty(2, 2).uniform_(2.0, 5.0)

# 正态分布 N(μ, σ²)
torch.empty(3, 3).normal_(mean=0.0, std=2.0)

# 设置随机种子（保证可复现）
torch.manual_seed(42)
```

---

## ✅ 小结

| 操作类型    | 函数                                                          |
| ------- | ----------------------------------------------------------- |
| 维度变换    | `view()`, `reshape()`                                       |
| 添加/去掉维度 | `unsqueeze()`, `squeeze()`                                  |
| 转置      | `t()`, `transpose()`, `permute()`                           |
| 扩展张量    | `expand()`, `repeat()`                                      |
| 随机初始化   | `rand()`, `randn()`, `randint()`, `uniform_()`, `normal_()` |

理解这些 Tensor 操作对图像处理、NLP、时间序列等领域都非常重要，能极大提升你操作数据的效率。