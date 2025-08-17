# 常用优化器类型及选择（optimizer）  -- 优化网络参数
## 常见优化器算法：
### 1. GD（梯度下降）
    1）包括三种常见的变形：批量梯段下降（BGD），随机梯度下降（SGD）使用最多，小批量梯度下降（MBGD）。区别在于每次计算梯度时选择样本的数量的方式不同
    2）学习率：所有优化器中算法中最重要的参数，直接影响梯度下降的过程（即步长，stride）、影响模型的学习速度（即收敛速度），过大可能导致在最小值附近来回震荡，无法收敛；国小则学习速度会过慢，初始设置为0.01~0.001为宜
### 2. Momentum（相比GD,增加了动量变化机制）
### 3. AdaGrad（自适应，相比GD,增加了学习率递减系数）
### 4. RMSProp（改良了AdaGrad，增加了衰减系数）
### 5. Adam（改良了RMSProp，增加了偏移校正） ，通用性最好，在不清楚选择时可以优先使用
后4种都是在GD算法基础上不断优化所得

# 常用损失函数的类型及选择（loss）
## 回归损失函数
### 1. mean_squared_error (mse) 均方误差，公式为：((y_pred-y_true)**2).mean()
### 2. mean_absolute_error (mae) 平均绝对误差，公式为：(|y_pred-y_true|).mean()
### 3. mean_absolute_percentage (mape) 平均绝对百分比误差
### 4. mean_squared_logarithmic_error (msle) 均方对数误差
## 分类损失函数
### 1. hinge：铰链损失函数，主要用于支持向量机（SVM）
### 2. binary_crossentropy: 二分类损失函数，交叉熵函数
### 3. categorical_crossentropy: 多酚类损失函数，交叉熵函数
### 4. sparse_categorical_crossentropy: 同上，多分类损失函数，可接受稀疏标签（无需转one-hot独热编码）


# 评价指标的选择（metrics） -- 即评估函数，用于计算模型的成绩，函数的输入为预测值和实际值，位于keras.metrics模块中
## 分类问题
### 1. binary_accuracy: 二分类问题，计算所有预测值上的平均正确率
### 2. categorical_accuracy: 多分类问题，计算所有预测值上的平均正确率
### 3. sparse_categorical_accuracy: 与categorical_accuracy相同，适用于稀疏标签预测
### 4. top_k_categorical_accuracy: 计算top-k正确率，当预测值的前k个值中存在目标类别即认为预测正确
### 5. sparse_top_k_categorical_accuracy： 与top_k_categorical_accuracy相同，适用于稀疏标签预测



# 训练时验证集参数的使用
## 训练集数据 x_train
## 训练集标签 y_train
## 训练迭代次数 epochs
## 批尺寸（每批次为给神经网络的样本数量） batch_size
    history = model.fit(x_train, y_train, epochs=1000, batch_size=800)
    训练时还可以指定验证数据，以更好地验证训练效果，两种方式：直接指定验证集，或从训练集中随机分割一定比例作为验证集
    history = model.fit(x_train, y_train, epochs=1000, batch_size=800, validation_data=(x_val,y_val))
    history = model.fit(x_train, y_train, epochs=1000, batch_size=800, validation_data=0.2)


# 卷积神经网络中池化的概念
    池化（也成为下采样或降采样）通过对特征图进行降维，达到减少参数数量、防止过拟合的目的。常用的池化类型有最大池化（Max Pooling）和平均池化（Average Pooling）
# 卷积层主要参数
    layers.Conv2D(filters,kernel_size,strides=(1,1),padding=‘valid’)
## filters 卷积核（过滤器）数量（即卷积后产生的特征图数量）
## kernel_size 卷积核尺寸（通常使用（3，3）或（5，5））
## strides 卷积步长，默认（1，1）
## padding 边界填充策略，默认valid表示不填充，卷积后特征图变小，same表示填充，卷积后与原图相同 
































