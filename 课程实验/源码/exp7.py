import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
import mindspore
from mindspore import Tensor, context, nn

# 设置 Mindspore 运行模式为图模式（graph mode）和设备为CPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 读取 Excel 文件
data = pd.read_excel('1.xls')

# 提取特征和目标变量
features = data[['人口数量', '机动车数量', '公路面积']].values
target = data[['公路客运量', '公路货运量']].values

# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 转换为 Mindspore Tensor 格式
X = Tensor(features_scaled, dtype=mindspore.float32)
y = Tensor(target, dtype=mindspore.float32)

class BPNeuralNetwork(nn.Cell):
    def __init__(self):
        super(BPNeuralNetwork, self).__init__()
        self.dense1 = nn.Dense(3, 16)  # 减少隐藏单元数量
        self.relu = nn.ReLU()
        self.dense2 = nn.Dense(16, 2)
    #定义了神经网络的前向计算过程
    def construct(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

# 初始化模型
model = BPNeuralNetwork()

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = nn.Adam(params=model.trainable_params(), learning_rate=3)

net_with_loss = nn.WithLossCell(model, loss_fn)

train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
train_network.set_train()

# 训练模型
epochs = 500
for epoch in range(epochs):
    loss = train_network(X, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.asnumpy()}")

# 2010年和2011年的特征数据
new_data = np.array([[73.39, 3.9635, 0.9880],  # 2010年数据
                     [75.55, 4.0975, 1.0268]])  # 2011年数据

# 标准化新数据
new_data_scaled = scaler.transform(new_data)

# 转换为 Mindspore Tensor 格式
new_data_tensor = Tensor(new_data_scaled, dtype=mindspore.float32)

# 预测
model.set_train(False)  # 设置模型为评估模式
predictions = model(new_data_tensor)

# 输出预测结果
print("2010及2011年预测值", predictions.asnumpy())

#线性回归模型
lr_model = sklearn.linear_model.LinearRegression()
lr_model.fit(features_scaled, target)

lr_predictions = lr_model.predict(new_data_scaled)
print("线性回归模型预测值", lr_predictions)