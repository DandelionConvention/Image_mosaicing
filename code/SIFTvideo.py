import cv2
import time

capture = cv2.VideoCapture(0)

fps = 0.0
while (True):
    t1 = time.time()
    # 读取某一帧
    ref, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # 找出关键点
    kp = sift.detect(gray, None)

    # 对关键点进行绘图
    ret = cv2.drawKeypoints(frame, kp, frame,color=(0,0,255))

    fps = (fps + (1. / (time.time() - t1 + 0.00001))) / 2
    print("fps= %.4f" % (fps))
    frame = cv2.putText(frame, "fps= %.4f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video", frame)
    c = cv2.waitKey(1) & 0xff

    if c == 27:
        capture.release()
        break

capture.release()
cv2.destroyAllWindows()