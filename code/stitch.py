# 导入必要的包
from panorama import Stitcher
import imutils
import cv2
import time

def equalize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    (H, L, S) = cv2.split(img)
    LH = cv2.equalizeHist(L)
    # 合并每一个通道
    result = cv2.merge((H, LH, S))
    result = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    return result

imageA = cv2.imread('./img/7.jpg')
imageB = cv2.imread('./img/07.jpg')
# cv2.imshow("Imag-A", imageA)
# cv2.imshow("Imag-B", imageB)
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
# 调整大小方便显示
imageB = equalize(imageB)
imageA = equalize(imageA)

s_t = time.time()

# 将图像拼接在一起以创建全景
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示图像
# cv2.imshow("Image A", imageA)
# cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
