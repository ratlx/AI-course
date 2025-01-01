import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mindspore import dataset as ds
from mindspore import nn, Tensor
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# 读取数据
data = pd.read_csv("iris.txt", delim_whitespace=True)
# 提取特征和标签
X = data[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].values
y = LabelEncoder().fit_transform(data["Species"].values)  # 转换类别为数字

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建一个简单的SVM分类器
class SVM(nn.Cell):
    def __init__(self, input_dim, num_classes):
        super(SVM, self).__init__()
        self.weight = Tensor(np.zeros((input_dim, num_classes)), dtype=np.float32)
        self.bias = Tensor(np.zeros(num_classes), dtype=np.float32)

    def construct(self, x):
        return x @ self.weight + self.bias

# 转换为Tensor格式
train_dataset = ds.NumpySlicesDataset({"features": X_train, "labels": y_train}, shuffle=True)
test_dataset = ds.NumpySlicesDataset({"features": X_test, "labels": y_test}, shuffle=False)

# 交叉验证部分（具体代码可根据MindSpore接口补充）
# 可用sklearn的交叉验证辅助实现：

svm_model = SVC(kernel="linear", C=1.0)
scores = cross_val_score(svm_model, X_train, y_train, cv=5)
print("交叉验证得分：", scores)

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 重新训练SVM模型
svm_model.fit(X_pca, y)
X_test_pca = pca.transform(X_test)

# 绘制分类边界
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

plot_decision_boundary(X_pca, y, svm_model)