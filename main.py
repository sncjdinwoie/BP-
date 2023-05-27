import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
# 训练数据集
df = pd.read_excel(r"C:\Users\bendaye\Desktop\机器学习数据.xlsx")
x = df[['x1', 'x2','x3']]
y = df[['y']]
# 归一化
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)
# 参数设置
epochs_max = 50000
learn_rate1 = 0.035
learn_rate2 = 0.05
hidden_layer = 5
error_min = 0.0001
# 初始化参数
w1 = 0.5*np.random.randn(len(x[0]), hidden_layer)-0.1
w2 = 0.5*np.random.randn(hidden_layer, len(y[0]))-0.1
b1 = 0.5*np.random.randn(hidden_layer)-0.1
b2 = 0.5*np.random.randn(len(y[0]))-0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    for epoch in range(epochs_max):
        # 前向传播
        hidden_output = sigmoid(np.dot(x, w1) - b1)
        predict_output = sigmoid(np.dot(hidden_output, w2) - b2)
        # 计算误差
        error = np.sum((predict_output - y) ** 2) / 2
        error = np.mean(error)
        if error < error_min:
            break
        # 反向传播
        g = predict_output * (1 - predict_output) * (y - predict_output)
        w2 += learn_rate2 * np.dot(hidden_output.T, g)
        b2 -= learn_rate2 * np.sum(g, axis=0)
        e = hidden_output * (1 - hidden_output) * np.dot(g, w2.T)
        w1 += learn_rate1 * np.dot(x.T, e)
        b1 -= learn_rate1 * np.sum(e, axis=0)
    # 反转获取真实值
    predict_output = y_scaler.inverse_transform(predict_output)
    sample_out = y_scaler.inverse_transform(y)
    # 图表
    plt.plot(predict_output, color="r", label="预测值")
    plt.plot(sample_out, color="blue", label="真实值")
    plt.legend()
    plt.show()