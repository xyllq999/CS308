import cv2
import numpy as np


def get_interest_points(image, feature_width):

    confidences, scales, orientations = None, None, None

    # 第一步，先实现哈里斯角点检测器
    # 参考https://www.cnblogs.com/zyly/p/9508131.html
    # k = 0.05
    # cv2.cornerHarris(image, 5, 3, k) 第三个参数代表敏感度，3-31的奇数，越大越敏感

    # 首先求xy方向的方向梯度，利用sobel算子，第一个参数-1，后面分别1，0和0，1
    # ix = cv2.Sobel(image, -1, 1, 0, ksize=3)
    # iy = cv2.Sobel(image, -1, 0, 1, ksize=3)
    # 注意如果sobel的size为3则可以使用Scharr算子
    ix = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    iy = cv2.Scharr(image, cv2.CV_64F, 0, 1)

    # 求Ix2，Iy2, IxIy
    ixx = np.square(ix)
    iyy = np.square(iy)
    ixy = np.multiply(ix, iy)

    # 使用高斯函数加权，sigma2 ksize3
    gaussian = cv2.getGaussianKernel(ksize=5, sigma=1.5)
    gauss_xx = cv2.filter2D(ixx, -1, gaussian)
    gauss_yy = cv2.filter2D(iyy, -1, gaussian)
    gauss_xy = cv2.filter2D(ixy, -1, gaussian)

    # R = det(M) - k(trace(M)) ** 2 k = 0.04 - 0.06
    # det(M) = xx * yy - xy ** 2
    # trace(M) = xx + yy
    k = 0.06
    det = np.multiply(gauss_xx, gauss_yy) - np.square(gauss_xy)
    trace = gauss_xx + gauss_yy
    R = det - k * np.square(trace)

    threshold = 10000
    n = 1500
    corners = []
    # 将每个点对应的值和坐标放进列表
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            corners.append([R[y, x], x, y])
    # 根据阈值并且降序排序,只保留最大的9000个点
    corners = np.array(sorted(corners, key=lambda x:x[0], reverse=True)[:threshold])

    # 把列表中的三个维度分别取出来
    x_list = corners[:, 1]
    y_list = corners[:, 2]

    # 接下来进行非极大值抑制，只要前1500个关键点
    # 采用了cs308-computer-Vision-master的非极大值抑制算法
    points = np.vstack([y_list, x_list]).T
    size = len(y_list)
    index = np.zeros(size)
    index[0] = np.inf
    # 单纯考虑距离的抑制，计算每个点到别的点的最近距离，最后将这些最近距离排序取前1500个
    for i in range(1, size):
        other_cordinate = i
        index[i] = np.min(np.sum(np.square(points[:other_cordinate] - points[i]), 1))

    # 参考https://blog.csdn.net/MacwinWin/article/details/80002584，使用argpatition用法
    # a[np.argpatition(a, -5)[-5:]] 获取前5大的
    # 获取topN的点
    x = np.array(x_list[np.argpartition(index, -n)[-n:]])
    y = np.array(y_list[np.argpartition(index, -n)[-n:]])

    return x, y, confidences, scales, orientations










