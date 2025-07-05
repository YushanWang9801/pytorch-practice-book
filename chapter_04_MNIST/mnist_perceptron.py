import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设置超参数
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # 标准化
    ]
)

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义感知机模型（输入层→输出层）
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)  # 784维 → 10类

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 拉平
        logits = self.linear(x)
        return logits  # 直接输出 logits（不加 softmax）


model = Perceptron().to(DEVICE)

# 损失函数 + 优化器
criterion = nn.CrossEntropyLoss()  # 包含 softmax + log
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


# 训练函数
def train():
    model.train()
    total_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()

        output = model(data)  # logits
        loss = criterion(output, target)  # CE Loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    acc = 100.0 * correct / len(train_loader.dataset)
    print(f"Train Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")


# 测试函数
def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    acc = 100.0 * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.2f}%")


import matplotlib.pyplot as plt


def visualize_predictions(n=10):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    plt.figure(figsize=(15, 4))
    for i in range(n):
        img = images[i].cpu().squeeze(0).numpy()
        true_label = labels[i].item()
        pred_label = preds[i].item()

        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap="gray")
        color = "green" if true_label == pred_label else "red"
        plt.title(f"P:{pred_label} / T:{true_label}", color=color)
        plt.axis("off")

    plt.suptitle("Sample Predictions (P: Predicted / T: True)", fontsize=16)
    plt.tight_layout()
    plt.show()


# 主训练循环
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}")
        train()
        test()

    visualize_predictions()
