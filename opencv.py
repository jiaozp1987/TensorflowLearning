import cv2
import os
import random
from keras.preprocessing import image
import numpy as np
from keras import layers, models
from keras.models import load_model



def data_capture():
    total = 200  # 默认采集的图片总数
    current = 0  # 当前已经采集的张数
    path = "./static/opencv"  # 图片存放的目录
    xml_path = "./.venv/Lib/site-packages/cv2/data/"
    prefix = 'nomask_'  # 图片前缀
    stop = True  # 当前是否是停止状态
    status = "Pause"  # 状态文字（Pause-暂停|Capturing-采集中|Finish-采集完成）
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义界面字体
    cap = cv2.VideoCapture(0)  # 开启摄像头
    face_cascade = cv2.CascadeClassifier(f"{xml_path}haarcascade_frontalface_alt2.xml")  # 创建人脸分类器（定位人脸）
    # 采集每一帧的人脸图片，保存
    while 1:
        ok, frame = cap.read()  # 读取当前帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰度
        #  检测当前帧中的人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))
        # 从faces中提取人脸，保存 x,y左上角坐标 w,h 长宽
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画矩形标记人脸,参数(帧，左上角坐标，右下角坐标，框的颜色，线的宽度)
            # 保存图片
            if not stop and (current < total):
                current = current + 1
                if current == total:
                    status = "Finish"
                # 图片路径和名称
                img_name = f"{path}/{prefix}{current}.jpg"
                face = gray[y:y + h, x:x + w]  # 人脸图片切片
                cv2.imwrite(img_name, face)

            # 显示当前已经抓取的图片数和状态
            cv2.putText(frame, f"{status}:{current}",(20,50),font,1,(0,255,0),2)


        # 显示当前帧
        cv2.imshow("Press ESC to Quit", frame)
        # 检测用户按键
        k = cv2.waitKey(1)
        if k == 27:  # ESC键，退出
            break
        if k == 32 and current<total:  # 空格键（暂停或继续）
            if stop :
                stop = False
                status = "Capturing"
            else:
                stop = True
                status = "Pause"
        if k == 13:  # 回车键 重置 重新抓取
            current = 0
            status = "Pause"
            stop = True
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def data_initialization():

    pic_size = 128  # 图片的长宽（pic_size，pic_size）
    pic_path = "./static/opencv"
    x_train,y_train,labels = preprocess(pic_path, pic_size)
    print(x_train.shape)
    print(y_train.shape)
    print(y_train)
    print(labels)
    # 使用matplotlib 查看图片
    # for i in range(10):
    #     plt.imshow(x_train[i].reshape(pic_size,pic_size), cmap="gray")
    #     plt.title(f"{labels[y_train[i]]}:{y_train[i]}")
    #     plt.show()
    return x_train,y_train,labels,pic_size

def preprocess(path,pic_size):
    """
    :description:  预处理函数
    :param :
    :return:
    """

    x , y = list(),list() # 特征数组，标签数组
    labels = list() # 标签名称
    cid = 0  # 当前图片的标签值
    files = os.listdir(path)
    random.shuffle(files)
    for file in files:
        file_name_items = file.split("_")
        label = file_name_items[0]
        if not label in labels:
            labels.append(label)
            cid = len(labels) - 1  # 当前图片的标签值
        y.append(labels.index(label)) # 将标签值加入标签数值
        # 获取当前图片的像素值数组
        src = f"{path}/{file}"
        img = image.load_img(src,color_mode="grayscale",target_size=(pic_size,pic_size)) # 加载 灰化 调整尺寸
        x_img = image.img_to_array(img)   # 转数组
        x_img = x_img.reshape(pic_size,pic_size,1)
        x.append(x_img)
    return np.array(x), np.array(y),np.array(labels)

def create_model(x_train,y_train,lables,pic_size):

    epochs = 10 # 迭代次数
    batch_size = 10 # 批大小
    classes = len(lables) # 类别的数量
    model = models.Sequential()
    # 第一组卷积
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(pic_size,pic_size,1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # 第二组卷积
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # 第三组卷积
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # 压平（转一维）
    model.add(layers.Flatten())

    # 全连接层做输出
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
    # 保存模型
    model.save('./static/opencv/my_face.h5')

def load_face_model(pic_size,labels):

    model = load_model('./static/opencv/my_face.h5')
    model.summary()
    xml_path = "./.venv/Lib/site-packages/cv2/data/"
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义界面字体
    cap = cv2.VideoCapture(0)  # 开启摄像头
    face_cascade = cv2.CascadeClassifier(f"{xml_path}haarcascade_frontalface_alt2.xml")  # 创建人脸分类器（定位人脸）
    while True:
        ok ,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰度
        #  检测当前帧中的人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))
        # 从faces中提取人脸，保存 x,y左上角坐标 w,h 长宽
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画矩形标记人脸,参数(帧，左上角坐标，右下角坐标，框的颜色，线的宽度)
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (pic_size, pic_size))
            face = np.array(face).reshape(-1,pic_size, pic_size,1) # 转成网络需要的形状
            result= model.predict(face)  # 调用模型进行预测
            index = np.argmax(result)
            name = labels[index]
            cv2.putText(frame,name,(x,y-10),font,1,(0,255,0),0) # 显示识别结果
        # 显示当前帧
        cv2.imshow("Press ESC to quit",frame)
        # 检测用户按键
        k = cv2.waitKey(1)
        if k == 27:  # ESC键，退出
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # data_capture()
    # x_train,y_train,labels,pic_size = data_initialization()
    # create_model(x_train,y_train,labels,pic_size)
    load_face_model(128,["mask","nomask"])
