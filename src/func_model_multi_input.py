# 搭建函数式多输入模型
import keras
import numpy as np
from keras import layers, models


def multi_input_model():
    # 定义参考文本、问题和答案的字典长度
    text_voc_size = 10000
    question_voc_size = 10000
    answer_voc_size = 500

    # 搭建“参考文本”输入分支
    text_input = layers.Input(shape=(None,), dtype='int32', name='text')
    embedded_text = layers.Embedding(text_voc_size, 64)(text_input)  # 嵌入层
    encoded_text = layers.LSTM(32)(embedded_text)  # 长短期记忆网络LSTM

    # 搭建“问题”输入分支
    question_input = layers.Input(shape=(None,), dtype='int32', name='question')
    embedded_question = layers.Embedding(question_voc_size, 32)(question_input)
    encoded_question = layers.LSTM(16)(embedded_question)

    # 搭建答案输出分支
    answer = layers.concatenate([encoded_text, encoded_question], axis=-1)

    # 定义输出层
    answer_out = layers.Dense(answer_voc_size, activation='softmax')(answer)

    # 创建模型
    model = models.Model([text_input, question_input], answer_out)
    model.summary()
    # 编译模型
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # 随机生成一些随机数据
    num_samples = 1000
    max_length = 100
    text_data = np.random.randint(1, text_voc_size, size=(num_samples, max_length))
    question_data = np.random.randint(1, question_voc_size, size=(num_samples, max_length))
    answer_targets = np.random.randint(answer_voc_size, size=(num_samples, 1))

    # 标签（answer）转独热编码（因为没使用sparse_categorical_crossentropy）
    answer_targets = keras.utils.to_categorical(answer_targets, answer_voc_size)

    # 训练模型
    model.fit([text_data, question_data], answer_targets, epochs=10, batch_size=128)
