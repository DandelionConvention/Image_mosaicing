# 导入必要的包
import numpy as np
import imutils
import cv2
from sklearn import linear_model

class Stitcher:
    def __init__(self):
        # 确定是否使用的是OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        # 解压缩图像，然后从它们中检测关键点以及提取局部不变描述符（SIFT）
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # print(kpsA.shape,featuresA.shape)
        # (325, 2) (325, 128)
        # 匹配两幅图像之间的特征

        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # 如果匹配结果M返回空，表示没有足够多的关键点匹配信息去创建一副全景图
        if M is None:
            return None

        # 若M不为None，则使用透视变换来拼接图像
        (matches, H, status) = M
        # print(np.array(matches).shape,np.array(H).shape,np.array(status).shape)
        # (74, 2) (3, 3) (74, 1)
        # matches对应kpsB和kpsA的索引,status对应是否匹配,匹配的话为1
        result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], np.max([imageA.shape[0],imageB.shape[0]])))

        # 调颜色
        if True:
            a_,b_ = self.adjustResult(result,imageB)
            b_ = b_ / 3
            result = a_ * result + b_
            result = np.where(result>255,255,result)
            result = np.where(result <= b_, 0, result)

            result = result.astype(np.uint8)

        for y in range(imageB.shape[0]):
            for x in range(imageB.shape[1]):
                if np.sum(result[y,x]) != 0:

                    result[y, x:x+5] = imageB[y, x:x+5]
                    result[y,x:imageB.shape[1]] = self.fusion(result[y,x:imageB.shape[1]],imageB[y,x:imageB.shape[1]],k=0.5)
                    break
                    #             遇到了变化后的A图，放在result的右边
                else:
                    result[y,x] = imageB[y,x]
        # result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检查是否应该可视化关键点匹配
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # 返回拼接图像的元组和可视化
            return (result, vis)

        # 返回拼接图像
        return result

    def adjustResult(self,res,img2,MeanF = True):
        # 调整饱和度和色调，计算重复区域
        over = np.zeros(img2.shape)
        X = []
        Y = []
        for y in range(img2.shape[0]):
            for x in range(img2.shape[1]):
                if np.sum(res[y, x]) != 0:
                    over[y, x] = 1
        # 切割后的重叠区域图
        imageoB = img2 * over
        reso = res[:img2.shape[0],:img2.shape[1]]
        # 作均值滤波
        imageoB = imageoB.astype(np.int16)
        reso = reso.astype(np.int16)
        if MeanF:
            imageoB = cv2.blur(imageoB,(5 ,5))
            reso = cv2.blur(reso, (5 ,5))

        for y in range(img2.shape[0]):
            for x in range(img2.shape[1]):
                if np.sum(res[y, x]) != 0:
                    X.append(np.sum(reso[y,x]))
                    Y.append(np.sum(imageoB[y,x]))

        X = np.array(X).reshape(-1,1)
        Y = np.array(Y).reshape(-1,1)

        model = linear_model.LinearRegression()
        model.fit(X, Y)
        b = model.intercept_.reshape(1)[0]
        a = model.coef_.reshape(1)[0]
        # 线性模型的系数
        print(a,b)
        return a,b



    def fusion(self,img1,img2,k=0.05):
        d = img1.shape[1]
        x = np.arange(-d / 2, d / 2)
        y = 1 / (1 + np.exp(-k * x))

        re = (1 - y) * img2 + y * img1
        return re


    def detectAndDescribe(self, image):
        # 将图像转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检查我们是否正在使用OpenCV 3.X
        if self.isv3:
            # 从图像中检测并提取特征
            # 这里的特征提取被cv封装了，如何进行特征提取很重要
            descriptor = cv2.xfeatures2d.SIFT_create()
            # orb = cv2.ORB_create()
            # surf = cv2.xfeatures2d.SURF_create()
            # 找到关键点和描述符

            (kps, features) = descriptor.detectAndCompute(image, None)
            # (kps, features) = surf.detectAndCompute(image, None)
            # (kps, features) = orb.detectAndCompute(image, None)


        # 否则，我们将使用OpenCV 2.4.X
        else:
            # 检测图像中的关键点
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # 从图像中提取特征
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # 将关键点从KeyPoint对象转换为NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回关键点和特征的元组
        # (325, 2) (325, 128)
        # 2是坐标值 128是特征值
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # 计算原始匹配项并初始化实际匹配项列表
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        # 更换匹配算法

        # 返回两个欧氏距离最近的点，只有当两个最近的点都小于某一值时，才算做匹配
        matches = []

        # 循环原始匹配
        for m in rawMatches:
            # 确保距离在一定的比例内(即Lowe's ratio)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        #         包含两幅图对应的坐标

        # 计算单应性至少需要4个匹配项
        if len(matches) > 4:
            # 构造两组点
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算两组点之间的单应性
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)
            # RANSAC    随机抽样一致
            # 返回匹配以及单应矩阵和每个匹配点的状态
            # H变换矩阵 status是排除了匹配错误的点
            return (matches, H, status)

        # 否则，将无法计算单应性
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

