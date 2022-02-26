import cv2
import numpy as np


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'

    k = len(x)
    feat_dim = 8 * 4 * 4
    fv = np.zeros((k, feat_dim))

    # feature_width为16，
    # print(image.shape)
    # 给图片边缘每一个方向增加8个像素宽度
    image = np.pad(image, feature_width // 2)
    # print(image.shape)

    # 算法思路来源于https://blog.csdn.net/weixin_38404120/article/details/73740612#commentBox
    # 经过harris角点检测算法之后，得到精确地特征点了，然后去求方向
    # 这里我取size为3所以使用更加高效的Scharr函数代替，将ddepth设置为cv2.CV_64F
    # ix = cv2.Sobel(image, -1, 1, 0, ksize=3)
    # iy = cv2.Sobel(image, -1, 0, 1, ksize=3)
    ix = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    iy = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    # print(ix)
    # print(iy)
    # 然后求mag(x,y)和theta(x,y) 即为像素对应的方向和大小构成直方图
    mag = np.sqrt(np.square(ix) + np.square(iy))
    theta = np.arctan2(iy, ix)

    # print(mag)
    # print(theta)
    # 只留下四个区间的值-1，0，1，2之后再进行+1得到0,1,2,3以及+4得到4，5，6，7正好加起来八个区间
    theta[theta > 1] = 2
    theta[theta < - 1] = -1

    # 需要将兴趣点位置转换为int然后对像素点进行操作
    x = x.astype(int)
    y = y.astype(int)

    # 对直方图进行分析,整个图分为16块，然后每一块把360度分为八个部分
    for index in range(k):
        histogram = np.zeros((4, 4, 8))
        for col in range(feature_width):
            for row in range(feature_width):
                curr = (y[index] + col, x[index] + row)
                # 如果遍历的点x偏导数大于0则在直方图上对应区域加上大小mag，在前四区
                if ix[curr] > 0:
                    histogram[col // 4, row // 4, int(np.ceil(theta[curr] + 1))] += mag[curr]
                # 如果不大于0同样的在对应直方图区域加上，在后四区
                else:
                    histogram[col // 4, row // 4, int(np.ceil(theta[curr] + 1 + 4))] += mag[curr]
        # 把每一个点对应的直方图转换为一维并且放入fv数组
        fv[index, :] = np.reshape(histogram, (1, 8 * 4 * 4))

    # 最后进行normalize, threshold过程，经过实验发现可以把正确率从0.78提升到0.88有明显的效果提升
    # 第一次normalize,变为单位向量
    # print(fv.shape[1])
    norm = np.sqrt(np.sum(np.power(fv, 2), 1))
    # print(norm)
    fv_norm = np.divide(fv, np.tile(norm, (128, 1)).T)
    # 多次试验出一个合适的阈值，最后发现0.1会导致match只能到95，0.40.50.6等等会使准确率下降到0.84甚至更低,最后发现最优的阈值0.2
    fv_norm[fv_norm > 0.2] = 0.2
    # 对fv向量进行再次标准化
    norm = np.sqrt(np.sum(np.power(fv_norm, 2), 1))
    # print(norm)
    fv = np.divide(fv_norm, np.tile(norm, (128, 1)).T)
    return fv
