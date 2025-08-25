import keras
import numpy as np
from keras import layers, models


def main():
    vocabulary_size = 500  # 帖子字典大小
    imcome_groups = 10  # 收入级别
    # 定义输入张量
    inputs = layers.Input(shape=(None,), dtype='int32', name='posts')
    # 定义嵌入层
    embedd_inputs = layers.Embedding(256, vocabulary_size)(inputs)
    # 定义卷积层
    x = layers.Conv1D(128, 5, activation='relu')(embedd_inputs)  # 处理文本用1维卷积
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Dense(128, activation='relu')(x)

    # 年龄输出
    age_prediction = layers.Dense(1, name='age')(x)
    # 收入输出
    income_prediction = layers.Dense(imcome_groups, activation='softmax', name='income')(x)
    # 性别输出
    gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

    # 创建模型
    model = models.Model(inputs=inputs, outputs=[age_prediction, income_prediction, gender_prediction])
    model.summary()

    # 为不同的输出指定不同的损失函数
    # model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
    # 也可以使用字典
    # model.compile(optimizer='adam',loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'})


    # 为不同输出的损失函数指定权重
    model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights=[0.25, 1, 10])