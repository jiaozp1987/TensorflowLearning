def main():
    import cv2
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

        # 显示当前帧
        cv2.imshow("Press ESC to Quit", frame)
        # 检测用户按键
        k = cv2.waitKey(1)
        if k == 27:  # ESC键，退出
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
