from keras import models, layers

"""
    SimpleRNN/LSTM层的常用参数。1-神经元个数，2-激活函数类型（默认为tanh）
    SimpleRNN因结构过于简单，实际任务中使用比较少，更多使用LSTM和GRU层
"""


def simple_rnn():
    model = models.Sequential()
    # 参数1、预处理获取的字典的大小（输入的维度）
    # 参数2、 输出的维度（相当于本层神经元的个数）
    model.add(layers.Embedding(1000, 32))  # Embedding层(嵌入层)的作用是将序列数据的整数下标向量化，只能作为网络的第一层
    model.add(layers.SimpleRNN(32, activation="tanh"))
    model.summary()


def lstm():
    model = models.Sequential()
    model.add(layers.Embedding(1000, 32))
    model.add(layers.LSTM(64,activation="tanh"))
    model.summary()


if __name__ == '__main__':
    simple_rnn()
    lstm()
