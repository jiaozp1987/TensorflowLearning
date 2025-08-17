import matplotlib.pyplot as plt
import numpy as np

# 前馈神经网络
def main1():
    from keras import datasets, layers, models
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 查看拆分结果
    print(x_train, y_train)
    print(x_test, y_test)
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(12, 4))
    # for i in range(10):
    #     ax = fig.add_subplot(2, 5, i + 1)
    #     ax.matshow(x_train[i])
    # 转换数据集的形状（转成二维）
    x_train = x_train.reshape(60000, 28 * 28)
    x_test = x_test.reshape(10000, 28 * 28)
    print(x_train.shape)
    # 搭建网络模型
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    # 显示网络概况
    model.summary()
    # 编译并且训练模型，评估准确率
    # 编译moxing
    # loss 如何计算网络误差
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(x_train, y_train, epochs=50, batch_size=28 * 28)
    # 在测试集上评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(test_acc)
    # 使用模型识别手写数字图片
    # 1） 识别测试集中的指定图片
    index = 9  # 指定待识别的图片序号
    plt.matshow(x_test[index].reshape(28, 28))  # 查看图片
    plt.show()
    # 使用模型识别该图片
    result = model.predict(x_test[index].reshape(-1, 28 * 28))
    result = np.around(result)
    print(result)
    print(np.argmax(result)) # 显示result中数字最大值的角标index、

    # 保存模型(权重和结构)
    model.save("mnist_model.h5")

    # 加载模型
    model = models.load_model("mnist_model.h5")


    # 保持模型(仅权重)
    model.save_weights('mnist_model_weights.h5')
    # 创建网络结构，，加载权重
    # 搭建网络结构
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    # 加载权重数据,加载后就可以直接使用模型
    model.load_weights('mnist_model_weights.h5')

    # 保存模型（仅保存网络结构）
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    model_json = model.to_json() # 把网络结构转json
    print(model_json)
    # 加载模型结构
    model = models.model_from_json(model_json)

# 卷积神经网络
def main2():
    pass

if __name__ == '__main__':
    # main1()
    pass
