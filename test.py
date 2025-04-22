import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===== 1. 簡単なNNを定義 =====
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# ===== 2. サンプル点を作成 =====
x = torch.linspace(0, 1, 50).view(-1, 1)  # (50点)
x.requires_grad_(True)

# ===== 3. 出力とパラメータ勾配を計算 =====
outputs = model(x)
params = list(model.parameters())
grads = []

for out in outputs:
    model.zero_grad()
    out.backward(retain_graph=True)
    grad = torch.cat([p.grad.flatten() for p in params])
    grads.append(grad.detach())

grads = torch.stack(grads)  # (50, パラメータ総数)

# ===== 4. NTK行列を作成 =====
NTK = grads @ grads.t()  # (50, 50)

# ===== 5. ヒートマップで可視化 =====
plt.figure(figsize=(6,5))
plt.imshow(NTK.numpy(), cmap="viridis")
plt.colorbar(label="NTK value")
plt.title("Neural Tangent Kernel (NTK) Matrix")
plt.xlabel("Sample index")
plt.ylabel("Sample index")
plt.show()
