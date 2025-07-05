import torch
import numpy as np

print("=" * 40)
print("📦 基本 Tensor 创建")
print("=" * 40)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
zeros = torch.zeros((2, 3))
ones = torch.ones((2, 2))
rand = torch.rand((2, 2))
eye = torch.eye(3)
like = torch.ones_like(x)

print(
    f"x:\n{x}\nzeros:\n{zeros}\nones:\n{ones}\nrand:\n{rand}\neye:\n{eye}\nlike:\n{like}"
)

print("=" * 40)
print("🔁 NumPy ↔ Tensor")
print("=" * 40)

a = np.array([[1, 2], [3, 4]])
t = torch.from_numpy(a)
a[0, 0] = 99
print("共享内存示例：", t)

t2 = torch.tensor([[5.0, 6.0]])
n = t2.numpy()
print("转换为 NumPy：", n)

print("=" * 40)
print("🧪 Tensor 基本操作")
print("=" * 40)

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print("加法:", x + y)
print("逐元素乘法:", x * y)
print("矩阵乘法:", torch.matmul(x, y))

print("=" * 40)
print("🔄 Tensor 维度变化")
print("=" * 40)

x = torch.arange(12).reshape(3, 4)
print("原始 x:\n", x)

x_reshaped = x.view(4, 3)
x_unsqueeze = x.unsqueeze(0)
x_squeezed = x_unsqueeze.squeeze()
print("reshape:", x_reshaped.shape)
print("unsqueeze:", x_unsqueeze.shape)
print("squeeze:", x_squeezed.shape)

print("=" * 40)
print("🧮 自动求导基础")
print("=" * 40)

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x + 1
y.backward()
print("dy/dx:", x.grad)  # 2x + 3 = 7

print("=" * 40)
print("⛓️ 多步链式反向传播")
print("=" * 40)

a = torch.tensor(1.0, requires_grad=True)
b = a * 2
c = b**2 + 3
c.backward()
print("dc/da =", a.grad)

print("=" * 40)
print("🎯 多变量自动求导（保留图）")
print("=" * 40)

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x * y + y**2
z.backward(retain_graph=True)  # 可重复 backward
print("dz/dx =", x.grad)
print("dz/dy =", y.grad)

# 再次 backward（梯度会累加）
z.backward()
print("再次 backward 后 dz/dx =", x.grad)

# 清除梯度
x.grad.zero_()
y.grad.zero_()

print("=" * 40)
print("📌 使用 torch.autograd.grad 手动获取梯度")
print("=" * 40)

x = torch.tensor(3.0, requires_grad=True)
y = x**3 + 2 * x
grad = torch.autograd.grad(y, x)
print("dy/dx:", grad)

print("=" * 40)
print("🧊 detach 和 requires_grad_ 的技巧")
print("=" * 40)

a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = a * 2
c = b.detach()  # 分离计算图
print("c 是否需要 grad：", c.requires_grad)

a.requires_grad_(False)
print("a.requires_grad=False:", a.requires_grad)

print("=" * 40)
print("🚀 GPU 可用性检测")
print("=" * 40)

print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
    t = torch.tensor([1.0, 2.0], device=device)
    print("当前设备:", t.device)
else:
    print("使用 CPU")
