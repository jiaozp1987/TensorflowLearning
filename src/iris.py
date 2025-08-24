from keras.models import Sequential
# 添加全连接层
from keras.layers import Dense
from keras import layers, models
# 导入数据集
from sklearn import datasets
# 导入数据预处理模块
from sklearn.model_selection import train_test_split
# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入评估模块
from sklearn.metrics import accuracy_score


def main():
    # # 加载数据集
    iris = datasets.load_iris()  # 加载数据集
    # print(iris.data)
    # print(iris.target)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)  # random_state=0 固定拆分结果
    # 搭建序贯式模型
    seq_model = Sequential()
    seq_model.add(Dense(units=32, activation='relu', input_dim=(4,)))
    seq_model.add(Dense(units=32, activation='relu'))
    seq_model.add(Dense(units=3, activation='softmax'))
    seq_model.summary()

    # 搭建函数式模型
    inputs = layers.Input(shape=(4,))  # 定义输入张量（tensor）
    x0 = layers.Dense(32, activation='relu')(inputs)
    x1 = layers.Dense(32, activation='relu')(x0)
    outputs = layers.Dense(3, activation='softmax')(x1)  # 定义输出张量（tensor）
    func_model = models.Model(inputs=inputs, outputs=outputs)  # 定义模型对象
    func_model.summary()
    # 训练序贯式模型
    seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    seq_model.fit(x_train, y_train, epochs=100)
    seq_model.evaluate(x_test, y_test)
    # 训练函数式模型
    func_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    func_model.fit(x_train, y_train, epochs=100)
    func_model.evaluate(x_test, y_test)
