import pandas as pd
from keras import layers, models
import matplotlib.pyplot as plt

def main():
    print("数据预处理开始")

    train_file = "../static/titanic/train.csv"

    df_train = pd.read_csv(train_file)

    # 取其中一部分列
    selected_cols = ['Survived', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df_train = df_train[selected_cols]

    df_train = df_train.sample(frac=1)  # 随机取样 1 表示全部取出即100% 就可以实现打乱顺序
    df_train = df_train.drop(['Name'], axis=1)  # 去掉姓名列
    age_mean = df_train['Age'].mean()
    df_train['Age'] = df_train['Age'].fillna(age_mean)  # 用平均值填报空值
    fare_mean = df_train['Fare'].mean()
    df_train['Fare'] = df_train['Fare'].fillna(fare_mean)  # 用平均值填报空值
    df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)  # 性别转数字
    df_train['Embarked'] = df_train['Embarked'].fillna("S")  # z众数填补
    df_train['Embarked'] = df_train['Embarked'].map({"C": 1, "Q": 2, "S": 3}).astype(int)  # 港口代码转数字

    ndarray_data = df_train.values  # 不包含列明的数据
    print(ndarray_data.shape)
    y_data = ndarray_data[:, 0]  # 标签数据
    features = ndarray_data[:, 1:]  # 特征数据
    from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化
    x_data = MinMaxScaler().fit_transform(features)  # 归一化
    # 拆分出训练集和测试集
    train_size = 800
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    x_test = x_data[train_size:]
    y_test = y_data[train_size:]
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print("数据预处理结束")
    print("搭建模型开始")

    model = models.Sequential()
    model.add(layers.Dense(512, activation="relu", input_shape=(7,)))  # 输入层
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(4, activation="relu"))
    model.add(layers.Dense(2, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))  # 输出层
    model.summary()
    # 模型编译
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    # 训练
    history = model.fit(x_train, y_train, epochs=1000, batch_size=800)
    # 训练过程可视化
    visu_history(history, "accuracy")
    visu_history(history, "loss")

    # 评估与预测
    from sklearn.metrics import classification_report
    # 在测试集上评估
    loss, accuracy = model.evaluate(x_test, y_test)
    # 因为是二分类，测试集预测出来的是概率, 设>0.5是1 否则为0
    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        y_pred[i] = 1 if y_pred[i] > 0.5 else 1

    cr = classification_report(y_test, y_pred)  # 生成分类评估报告
    print(cr)

    print("搭建模型结束")


# 训练过程可视化
def visu_history(history, metric_name):

    plt.title("Train history")
    plt.plot(history.history[metric_name])
    plt.xlabel('opochs')
    plt.ylabel(metric_name)
    plt.show()


if __name__ == '__main__':
    main()
