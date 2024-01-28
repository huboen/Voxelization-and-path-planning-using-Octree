import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_size,hidden_size,output_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    


# 生成虚拟数据集
def generate_data(num_samples, input_size, output_size):
    inputs = torch.randn(num_samples, input_size)
    targets = torch.randint(0, output_size, (num_samples,))
    return inputs, targets

# 定义模型和数据集维度
input_size = 10
hidden_sizes = 20
output_size = 5

# 创建多隐藏层神经网络
model = Actor(input_size, hidden_sizes, output_size)

# 生成数据集
num_samples = 1000
inputs, targets = generate_data(num_samples, input_size, output_size)

# 划分数据集为训练集和测试集
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_dataset = TensorDataset(inputs[:train_size], targets[:train_size])
test_dataset = TensorDataset(inputs[train_size:], targets[train_size:])

# 使用DataLoader加载数据
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    for inputs_batch, targets_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        loss.backward()
        optimizer.step()

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs_batch, targets_batch in test_loader:
            outputs = model(inputs_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += targets_batch.size(0)
            correct += (predicted == targets_batch).sum().item()

    accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2%}')

# 在测试集上进行预测
test_inputs, test_targets = generate_data(10, input_size, output_size)
with torch.no_grad():
    test_outputs = model(test_inputs)
    _, predicted_classes = torch.max(test_outputs, 1)

print("Predicted Classes:", predicted_classes)