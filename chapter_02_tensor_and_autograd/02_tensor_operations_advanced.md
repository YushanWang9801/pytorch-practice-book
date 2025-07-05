# 附录：PyTorch 中常用的高级 Tensor 操作

本节整理了 PyTorch 中常用的 Tensor 合并、数学运算、近似函数、裁剪、统计与高级索引操作。

---

## 🔗 Merge / Split 操作

### `torch.cat`（按某维拼接）
```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
out = torch.cat((a, b), dim=0)  # shape: (4, 3)
```

### `torch.stack`（增加新维度后拼接）

```python
a = torch.randn(3)
b = torch.randn(3)
out = torch.stack((a, b), dim=0)  # shape: (2, 3)
```

### `torch.split`（按指定大小切分）

```python
x = torch.arange(10)
a, b = torch.split(x, 5)  # 每段长度为5
```

### `torch.chunk`（按段数切分）

```python
x = torch.arange(12)
chunks = torch.chunk(x, 3)  # 分成3段（不一定均匀）
```

---

## ➗ 数学运算

### 基本运算

```python
a + b       # 加法
a - b       # 减法
a * b       # 元素乘
a / b       # 元素除
```

### 矩阵运算

```python
torch.matmul(a, b)   # 矩阵乘法
```

### 幂与根

```python
torch.pow(a, 2)      # a 的平方
torch.sqrt(a)        # 平方根
torch.rsqrt(a)       # 倒数开方：1/sqrt(a)
```

### 取整

```python
torch.round(x)
torch.floor(x)
torch.ceil(x)
torch.trunc(x)   # 向0截断（保留整数部分）
torch.frac(x)    # 取小数部分（x - trunc(x)）
```

---

## 🔒 clamp 操作

用于限制张量的范围（常用于梯度裁剪、激活裁剪等）

### 1. 基本使用

```python
x = torch.tensor([-3.0, 0.5, 10.0])
y = torch.clamp(x, min=0.0, max=5.0)  # [0.0, 0.5, 5.0]
```

### 2. 仅限制最小值

```python
torch.clamp_min(x, 0.0)
```

### 3. 用于梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 📊 常用统计函数

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
```

### 聚合运算

```python
x.sum()         # 所有元素求和
x.mean()        # 平均值
x.prod()        # 连乘
```

### 最大/最小

```python
x.max()         # 最大值
x.min()         # 最小值
x.argmax()      # 最大值的索引
x.argmin()      # 最小值的索引
```

### 排序相关

```python
torch.kthvalue(x.flatten(), k=2)   # 第2小的值
torch.topk(x.flatten(), k=2)       # Top-2 最大值
```

---

## 🧮 Norm 相关

### 向量范数（默认 p=2，即 L2）

```python
v = torch.tensor([3.0, 4.0])
torch.norm(v)   # = sqrt(3^2 + 4^2) = 5.0
```

### 矩阵范数（可指定维度）

```python
m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
torch.norm(m, p='fro')  # Frobenius norm
```

### 与 `normalize` 的区别

* `torch.norm()` 计算范数的**值**
* `F.normalize()` 返回**归一化后的向量**

```python
import torch.nn.functional as F
F.normalize(v, p=2, dim=0)
```

### batch norm / layer norm 等是网络层的一部分，不属于张量基础操作。

---

## 🧠 高级 Tensor 操作

### `torch.where(condition, x, y)`

三元选择：如果 condition 为 True 选 x，否则选 y。

```python
x = torch.tensor([1.0, -1.0, 3.0])
y = torch.where(x > 0, x, torch.zeros_like(x))  # 保留正数，其它为 0
```

### `torch.gather(input, dim, index)`

沿指定维度按 `index` 采样，功能强于普通索引，适合 batch 处理。

```python
x = torch.tensor([[10, 20, 30], [40, 50, 60]])
idx = torch.tensor([[2], [1]])
g = torch.gather(x, dim=1, index=idx)  # [[30], [50]]
```

---

## ✅ 小结

| 类型    | 常用操作                                      |
| ----- | ----------------------------------------- |
| 拼接/切分 | `cat`, `stack`, `split`, `chunk`          |
| 数学    | `+`, `*`, `/`, `matmul`, `pow`, `sqrt`    |
| 近似    | `round`, `floor`, `ceil`, `trunc`, `frac` |
| 限制    | `clamp`, `clip_grad_norm_`                |
| 统计    | `sum`, `mean`, `prod`, `argmax`, `topk`   |
| 范数    | `norm`, `normalize`                       |
| 选择    | `where`, `gather`                         |
