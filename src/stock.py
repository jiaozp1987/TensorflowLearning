"""
使用长短记忆网络（LSTM）建立股票收盘价预测模型
"""

import pandas as pd

from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def main():
    df = pd.read_csv("../static/stock.csv")   # 读取股票数据
    # 列名 Date   Open    High    Low     Close   Adj Close   Volume  共7列
    # 数据可视化
    # 将Date列转成日期，并设为索引（便于可视化）
    df["Date"] = pd.to_datetime(df["Date"],format="%Y-%m-%d")
    df.index = df["Date"]
    # 画图
    plt.figure(figsize=(15,6))
    plt.title("stock price")
    plt.plot(df["Close"],label="Close Price")
    plt.legend()
    plt.grid()
    plt.show()
    # 数据预处理
    # 取出收盘价进行数据归一化
    df_close = df["Close"]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df_close.values.reshape(-1,1))
    time_steps = 60 # 时间步长，相当于使用序列中前60个值预测下一个值（标签）
    train_size = 1000

    # 拆分训练集（数据、标签）和测试集
    x_train,y_train,x_test = list(),list(),list()
    for i in range(time_steps,len(data)):
        if i < train_size:
            x_train.append(data[i-time_steps:i,0])
            y_train.append(data[i,0])
        else:
            x_test.append(data[i-time_steps:i,0])
    x_train,y_train,x_test = np.array(x_train),np.array(y_train),np.array(x_test)
    print(x_train.shape,y_train.shape,x_test.shape)
    # 转成LSTM网络所需的输入维度（三维）
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1)) # 与 x_train.reshape(x_train.shape[0],x_train.shape[1],1) 是一样的
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    print(x_train.shape, y_train.shape, x_test.shape)
    # 建立LSTM网络，编译并训练
    epochs = 5
    batch_size = 10
    model = models.Sequential()
    # 只有 return_sequences = True 使该层输出为序列， 接下来后面才能继续加LSTM层，
    model.add(layers.LSTM(50,return_sequences= True,input_shape=(x_train.shape[1],1)))
    model.add(layers.LSTM(50))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer='adam',loss='mean_squared_error')
    history = model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)

    # 训练过程指标可视化
    plt.title("Training loss curve")
    plt.plot(range(epochs), history.history["loss"], label="loss")  # 误差变化曲线
    plt.legend()
    plt.grid()
    plt.show()

    # 预测测试集股价走势，并与实际走势进行对比
    y_pred = model.predict(x_test)
    # x_test已经归一化了，y_pred需要反归一化
    y_pred = scaler.inverse_transform(y_pred) # 反归一化，恢复量杠
    # 可视化
    train = df[train_size]
    test = df[train_size:]
    test["Pred"] = y_pred
    plt.figure(figsize=(15,6))
    plt.title("stock price",fontsize=20)
    plt.plot(train["Close"],label="Train_set Real Price")  # 训练集实际走势
    plt.plot(test["Close"], label="Test_set Real Price",color="green") # 测试集实际走势
    plt.plot(train["Pred"], label="Test_set Pred Price",color="red") # 测试集预测走势
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()