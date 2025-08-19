"""
1. 加载MDB数据集
    使用imdb.load_data() 函数加载，第一次加载需下载
    也可以自己从网上下载数据集文件imdb.npz 放在当前用户目录下的.keras/datasets目录下
    imdb数据集已经将单词进行了编号，num_words参数指定加载出现频率最高的前多少个词，舍弃剩下的低频词，降低向量的大小
2. 将平路序列转为定长序列
    因为数据集中每条评论的单词个数不同，需要转换成定长序列
    使用sequence.pad_sequences函数将评论转成定长序列，参数maxlen指定长度（超过则截取，，不足则补0）
3. 搭建循环神经网络
    本任务属于二分类问题，所以输出层神经元个数为1，且使用sigmoid激活函数
    二分类问题，损失函数loss使用binary_crossentropy(二分类交叉熵函数)
    嵌入层（Embedding）将序列的整数下标转为向量，该层只能用作网络的第一层
    长短记忆网络LSTM比SimpleRNN更常用，再本例中测试集评估效果也更好一些
4.将训练过程可视化并在测试集上进行评估
    利用fit方法返回训练过程数据,使用Matplotlib进行可视化
5. 预测测试机中的评论
"""

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence  # 序列数据的预处理
from keras import layers, models
import matplotlib.pyplot as plt

num_words = 10000
maxlen = 300  # 定长长度，尽量覆盖所有评论的长度
epochs = 5  # 迭代训练此时
batch_size = 128  # 批大小


def main():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    # 查看数据集形状
    print(x_train.shpe, x_test.shape)
    # 查看一条评论数据
    print(x_train[0])
    print(y_train[0])

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    print(x_train.shape, x_test.shape)

    model = models.Sequential()
    model.add(layers.Embedding(num_words, 64))
    model.add(layers.LSTM(16))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)  # 评估成绩

    # 训练过程指标可视化
    plt.title("Training metrics curve")
    plt.plot(range(epochs), history.history["acc"], label="acc")  # 准确率变化曲线
    plt.plot(range(epochs), history.history["loss"], label="loss")  # 误差变化曲线
    plt.legend()
    plt.grid()
    plt.show()

    # 预测
    num = 30  # 预测测试集前30条数据
    pred = model.predict(x_test[:num])
    # 将输出结果转为（0，1）值
    y_pred = list()
    for i in range(num):
        y_pred.append(0 if pred[i] <= 0.5 else 1)
    print(f"预测结果,{np.array(y_pred)}")
    print(f"实际结果,{y_test[:num]}")


if __name__ == '__main__':
    main()
