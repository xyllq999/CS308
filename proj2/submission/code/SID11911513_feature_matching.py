import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    image1_num = len(x1)
    image2_num = len(x2)
    max_num = 100

    matched_points = []
    # 接下来进行ratio test
    for index in range(image1_num):
        # 首先将feature1中某一个点拓展成和feature2一样长度方便运算
        features1_extend = np.tile(features1[index, :], (image2_num, 1))
        # 然后计算到feature2中每一个点的距离
        distance = np.sum(np.square(features1_extend - features2), 1)
        # 找到最近的两个距离的下标，也就是feature1当前点距离feature2最近的两个点的下标
        min_dis_index = np.argpartition(distance, 2)[:2]
        ratio = distance[min_dis_index[0]] / distance[min_dis_index[1]]
        # 设置阈值为0.2
        if ratio < 0.6:
            # 如果阈值小于0.6则加入ratio,f1坐标，f2最近的点坐标
            matched_points.append([ratio, index, min_dis_index[0]])

    # 对列表根据ratio进行排序
    sorted_matches = sorted(matched_points, key=lambda x: x[0])
    # 获得最好的100个匹配点对, 并且保留f1和f2的坐标
    matches = np.array([[point[1], point[2]] for point in sorted_matches[:max_num]]).astype(int)
    # 将ratio存入confidences数组
    confidences = np.array(point[0] for point in sorted_matches[:max_num])

    return matches, confidences
