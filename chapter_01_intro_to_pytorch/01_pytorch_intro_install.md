# 第 1 章：PyTorch 简介与环境配置

本章将带你完成以下内容：

- 什么是 PyTorch？
- 安装 PyTorch 的推荐方式
- 虚拟环境的配置（推荐使用 Conda）
- CPU 与 GPU 的区别
- 安装 GPU 驱动 + CUDA + cuDNN 的建议
- 如何测试 PyTorch 是否能正确使用 GPU

---

## 🔍 什么是 PyTorch？

PyTorch 是一个由 Facebook 开发的深度学习框架，具有以下优点：

- **动态计算图**：更灵活、易于调试
- **Python 原生支持**：符合 Python 编程习惯
- **强大的生态系统**：TorchVision, TorchText, TorchAudio 等
- **广泛的社区支持**：官方文档丰富，开源项目众多

---

## 🛠️ 安装推荐：使用 Conda 创建虚拟环境

我们建议使用 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 来管理 Python 环境。

### 创建新的虚拟环境

```bash
conda create -n pytorch-tutorial python=3.10
conda activate pytorch-tutorial
```

### 安装 PyTorch（根据是否使用 GPU 区分）

可以访问 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 获取最新的安装命令。

#### ✅ 安装 CPU 版本（通用）

```bash
pip install torch torchvision torchaudio
```

#### ✅ 安装 GPU 版本（以 CUDA 11.8 为例）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果你使用的是 Conda，可以：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## ⚙️ CPU 与 GPU 的区别

| 对比项 | CPU           | GPU            |
| --- | ------------- | -------------- |
| 优势  | 控制逻辑强、适合通用计算  | 并行能力强、适合矩阵运算   |
| 缺点  | 慢，无法加速大规模神经网络 | 需要安装驱动、CUDA 支持 |
| 场景  | 学习、调试、部署小模型   | 训练大模型、图像/视频任务  |

---

## 💡 如何配置 GPU 支持（NVIDIA 系列）

> 若你没有 NVIDIA 显卡，可以跳过本节。

### 1. 检查你的 NVIDIA 显卡

```bash
nvidia-smi
```

如果命令不存在或报错，请先安装 [NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx)。

### 2. 安装 CUDA（推荐和 PyTorch 对应的版本）

访问：[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

### 3. 安装 cuDNN（选装）

访问：[cuDNN 下载页面](https://developer.nvidia.com/cudnn)，解压到对应的 CUDA 路径下。

**注意**：conda 安装 `pytorch-cuda` 时已内置 cuDNN，无需手动配置。

---

## 🧪 测试 PyTorch 是否成功安装（及 GPU 可用性）

进入 Python 环境并执行：

```python
import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("当前 GPU:", torch.cuda.get_device_name(0))
    print("GPU 数量:", torch.cuda.device_count())
```

---

## 📦 附加建议

* 使用 `requirements.txt` 或 `environment.yml` 管理依赖
* 将项目组织成模块，便于后续实验和复现
* 推荐 IDE：VS Code + Python 插件 + Jupyter 插件

---

## ✅ 小结

你现在应该已经：

* 知道了 PyTorch 的基本用途
* 创建了一个干净的虚拟环境
* 成功安装了 PyTorch
* 验证了是否可以使用 GPU