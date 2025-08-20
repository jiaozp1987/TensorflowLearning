from keras import models, layers, datasets
from keras.callbacks import TensorBoard


def main():
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 转换数据集的形状（转成二维）
    x_train = x_train.reshape(60000, 28 * 28)
    x_test = x_test.reshape(10000, 28 * 28)

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 指定训练日志的目录
    log_dir = "../static/logs"

    # 定义TensorBoard对象
    tensor_boards = TensorBoard(log_dir=log_dir, histogram_freq=1)  # histogram_fraq=1 为统计直方图
    model.fit(x_train, y_train, epochs=50, batch_size=28 * 28, validation_split=0.1, callbacks=[tensor_boards])

    # logs文件夹里已有日志
    # 启动见README  10.1

if __name__ == '__main__':
    main()
