import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, context, ops

# 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 1. 导入数据并进行归一化
def load_and_normalize_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :2]  # 面积和房间数
    y = data[:, 2]   # 房价
    mu = np.mean(X, axis=0)  # 计算均值
    sigma = np.std(X, axis=0)  # 计算标准差
    y_max = np.max(y)
    y = y / y_max  # 缩放 y
    X_normalized = (X - mu) / sigma  # 均值归一化
    return X_normalized, y, mu, sigma, y_max

X, y, mu, sigma, y_max = load_and_normalize_data("ex1data2.txt")

# 添加偏置项
X = np.c_[np.ones(X.shape[0]), X]
y = y.reshape(-1, 1)

# 转换为 Tensor 格式
X = Tensor(X, ms.float32)
y = Tensor(y, ms.float32)

# 2. 实现线性回归模型
class LinearRegression(nn.Cell):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.theta = ms.Parameter(ops.Zeros()((input_dim, 1), ms.float32))

    def construct(self, X):
        return ops.matmul(X, self.theta)

# 定义损失函数
def compute_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = (1 / (2 * m)) * ops.reduce_mean((y_pred - y_true) ** 2)
    return loss

# 3. 梯度下降
def gradient_descent(model, X, y, alpha, num_iters):
    m = y.shape[0]
    optimizer = nn.SGD(params=[model.theta], learning_rate=alpha)
    grad_fn = ops.GradOperation(get_by_list=True)  # 获取梯度计算函数

    for _ in range(num_iters):
        def forward_fn():
            loss = compute_loss(model(X), y)
            return loss

        # 计算梯度
        grads = grad_fn(forward_fn, model.trainable_params())()
        optimizer(grads)
    return model.theta

# 尝试不同的 alpha
alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
num_iters = 400
best_alpha = 0
min_loss = float('inf')

for alpha in alphas:
    model = LinearRegression(X.shape[1])
    theta = gradient_descent(model, X, y, alpha, num_iters)
    loss = compute_loss(model(X), y)
    if loss < min_loss:
        min_loss = loss
        best_alpha = alpha

print(f"最佳学习率: {best_alpha}, 最小损失: {min_loss}")

# 使用最佳 alpha 训练模型
model = LinearRegression(X.shape[1])
theta = gradient_descent(model, X, y, best_alpha, num_iters)

# 4. 预测房价
def predict(model, mu, sigma, features):
    features_normalized = (features - mu) / sigma
    features_normalized = np.insert(features_normalized, 0, 1)  # 添加偏置
    features_normalized = Tensor(features_normalized, ms.float32)
    return ops.matmul(features_normalized, model.theta)

features = np.array([1650, 3])
predicted_price = predict(model, mu, sigma, features) * y_max
print(f"预测房价: {predicted_price.asnumpy()[0]:.2f}")