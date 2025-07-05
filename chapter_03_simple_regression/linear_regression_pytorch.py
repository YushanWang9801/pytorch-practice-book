import torch
import matplotlib.pyplot as plt

# 自动选择 CPU 或 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_data(file_path):
    data = torch.tensor(
        [list(map(float, line.strip().split(","))) for line in open(file_path)],
        dtype=torch.float32,
    ).to(device)
    x = data[:, 0].unsqueeze(1)  # shape: (N, 1)
    y = data[:, 1].unsqueeze(1)
    return x, y


def train(x, y, learning_rate=1e-4, num_epochs=1000):
    # 初始化参数（可学习）
    w = torch.randn(1, 1, requires_grad=True, device=device)
    b = torch.zeros(1, requires_grad=True, device=device)

    optimizer = torch.optim.SGD([w, b], lr=learning_rate)

    for epoch in range(num_epochs):
        # 前向传播
        y_pred = x @ w + b

        # MSE Loss
        loss = torch.mean((y - y_pred) ** 2)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch:4d}: Loss = {loss.item():.6f}, w = {w.item():.4f}, b = {b.item():.4f}"
            )

    return w.detach(), b.detach()


def plot_fit(x, y, w, b):
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    y_pred = x_cpu @ w.cpu() + b.cpu()

    plt.scatter(x_cpu.numpy(), y_cpu.numpy(), label="Data")
    plt.plot(x_cpu.numpy(), y_pred.numpy(), color="red", label="Fitted Line")
    plt.legend()
    plt.title("Linear Regression with PyTorch")
    plt.show()


def main():
    x, y = load_data("data.csv")
    w, b = train(x, y, learning_rate=1e-4, num_epochs=1000)
    print(f"\nTrained Model: y = {w.item():.4f} * x + {b.item():.4f}")
    plot_fit(x, y, w, b)


if __name__ == "__main__":
    main()
