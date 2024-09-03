import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# pandas: 用于加载和操作数据。
# torch: PyTorch库，用于构建和训练神经网络。
# torch.nn: 包含构建神经网络所需的模块和类。
# torch.optim: 包含优化算法，如Adam。
# DataLoader 和 TensorDataset: 用于创建和管理数据加载器。
# MinMaxScaler: 用于数据标准化，将数据缩放到指定的范围内。



# 加载数据
data = pd.read_csv('data/datatransfer/time_series_data.csv')
data = data[['RSRP', 'RSRQ', 'SINR', 'user_count', 'traffic_data']]  # 选择相关列

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
data_scaled = torch.tensor(data_scaled, dtype=torch.float32)

# 首先从CSV文件加载时序数据，并选择相关的列。
# 使用 MinMaxScaler 对数据进行标准化，使得每个特征的值范围在0到1之间，有助于模型训练的稳定性和收敛速度。
# 将标准化后的数据转换为PyTorch张量。


# 创建数据集
def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back):
        dataX.append(data[i:(i+look_back), :])
        dataY.append(data[i + look_back, :])
    return torch.stack(dataX), torch.stack(dataY)

look_back = 1
train_size = int(len(data_scaled) * 0.8)
trainX, trainY = create_dataset(data_scaled[:train_size], look_back)
testX, testY = create_dataset(data_scaled[train_size:], look_back)

# create_dataset 函数用于将时序数据转换为机器学习模型可用的输入和输出格式。look_back 参数决定了输入数据的时间窗口大小。
# 数据被分割为训练集和测试集，这里使用了80%的数据作为训练集。


# 创建DataLoaders
train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=1, shuffle=True)
test_loader = DataLoader(TensorDataset(testX, testY), batch_size=1, shuffle=False)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))


# LSTMModel 是一个基于PyTorch的神经网络模型，包含LSTM层和全连接层。
# forward 方法定义了数据如何通过网络传递。
# init_hidden 方法初始化隐藏状态和细胞状态。

# 实例化和训练模型
model = LSTMModel(input_dim=5, hidden_dim=50, num_layers=1, output_dim=5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # 实例化模型并设置损失函数和优化器。
    # 进行100轮训练，每轮通过所有的训练数据。对于每个批次，计算预测值，损失，并通过反向传播更新模型权重。