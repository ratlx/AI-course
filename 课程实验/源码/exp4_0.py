import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mindspore import Tensor, nn
import mindspore as ms
import mindspore.ops as ops

# 导入数据
data = pd.read_csv('ex2data1.txt', header=None, names=['Exam1', 'Exam2', 'Admitted'])

# 查看前几行数据
print(data.head())

# 绘制数据分布
plt.figure(figsize=(8, 6))
plt.scatter(data[data['Admitted'] == 1]['Exam1'],
            data[data['Admitted'] == 1]['Exam2'],
            c='b', marker='x', label='Admitted')
plt.scatter(data[data['Admitted'] == 0]['Exam1'],
            data[data['Admitted'] == 0]['Exam2'],
            c='r', marker='o', label='Not Admitted')
plt.title("Student Admission Based on Exam Scores")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

# 特征和标签
X = data[['Exam1', 'Exam2']].values
y = data['Admitted'].values

# 添加偏置项
X = np.insert(X, 0, 1, axis=1)

# 初始化参数
theta = np.zeros(X.shape[1])

# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义代价函数
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5  # 避免log(0)
    cost = (-1 / m) * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
    return cost

# 初始代价
initial_cost = cost_function(theta, X, y)
print(f"Initial Cost: {initial_cost}")

def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * np.dot(X.T, (h - y))
    return grad

# 使用Mindspore优化器
class LogisticRegressionModel(nn.Cell):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Dense(input_size, 1, weight_init='zeros', bias_init='zeros')

    def construct(self, x):
        return ops.sigmoid(self.linear(x))

# 数据预处理
X_tensor = Tensor(X, dtype=ms.float32)
y_tensor = Tensor(y.reshape(-1, 1), dtype=ms.float32)

# 创建模型
model = LogisticRegressionModel(X_tensor.shape[1])
loss_fn = nn.BCELoss()  # 二分类交叉熵损失
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)

# 使用Mindspore进行训练
model_with_loss = nn.WithLossCell(model, loss_fn)
train_net = nn.TrainOneStepCell(model_with_loss, optimizer)

# 训练
epochs = 1000
for epoch in range(epochs):
    train_loss = train_net(X_tensor, y_tensor)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {train_loss}")

student_score = np.array([1, 45, 85])  # 包含偏置项
student_score_tensor = Tensor(student_score, dtype=ms.float32)

# 预测
probability = model(student_score_tensor)
print(f"Predicted Probability: {probability.asnumpy()[0]}")

# 绘制分类边界
coef = model.linear.weight.asnumpy().flatten()
intercept = model.linear.bias.asnumpy()[0]

x_values = np.linspace(data['Exam1'].min(), data['Exam1'].max(), 100)
y_values = -(intercept + coef[0] + coef[1] * x_values) / coef[2]

plt.figure(figsize=(8, 6))
plt.scatter(data[data['Admitted'] == 1]['Exam1'],
            data[data['Admitted'] == 1]['Exam2'],
            c='b', marker='x', label='Admitted')
plt.scatter(data[data['Admitted'] == 0]['Exam1'],
            data[data['Admitted'] == 0]['Exam2'],
            c='r', marker='o', label='Not Admitted')
plt.plot(x_values, y_values, 'g-', label='Decision Boundary')
plt.title("Student Admission Based on Exam Scores")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()