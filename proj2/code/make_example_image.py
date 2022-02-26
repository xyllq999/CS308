import matplotlib.pyplot as plt
import time
from utils import *
from SID11911513_feature_matching import match_features
from SID11911513_sift import get_features
from SID11911513_harris import get_interest_points

# 保持和proj2中一致的参数
scale_factor = 0.5
feature_width = 16

images = [
    ['../data/Capricho Gaudi/test1a.jpg',
     '../data/Capricho Gaudi/test1b.jpg'],
    ['../data/Mount Rushmore/test2a.jpg',
     '../data/Mount Rushmore/test2b.jpg'],
    ['../data/Episcopal Gaudi/test3a.jpg',
     '../data/Episcopal Gaudi/test3b.jpg'],
    ['../data/House/test4a.JPG',
     '../data/House/test4b.JPG'],
    ['../data/Sacre Coeur/test5a.jpg',
     '../data/Sacre Coeur/test5b.jpg']
]

for index, images in enumerate(images):
    image1 = load_image(images[0])
    image2 = load_image(images[1])

    print(f"\n开始匹配第{index}组图像")
    start_time = time.time()

    image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
    image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    x1, y1, _, _, _ = get_interest_points(image1_gray, feature_width)
    x2, y2, _, _, _ = get_interest_points(image2_gray, feature_width)

    print('在image1发现了{:d}个corners, 在image2发现了{:d}个corners'.format(len(x1), len(x2)))

    image1_features = get_features(image1_gray, x1, y1, feature_width)
    image2_features = get_features(image2_gray, x2, y2, feature_width)

    matches_point, _ = match_features(image1_features, image2_features, x1, y1, x2, y2)
    print('{:d} corners 匹配到了 {:d} 个点'.format(len(x1), len(matches_point)))
    print(f"共耗时：{time.time() - start_time}")

    max_to_show = 100
    circle_image = show_correspondence_circles(image1, image2,
                                     x1[matches_point[:max_to_show, 0]],
                                     y1[matches_point[:max_to_show, 0]],
                                     x2[matches_point[:max_to_show, 1]],
                                     y2[matches_point[:max_to_show, 1]])
    plt.figure()
    plt.imshow(circle_image)
    plt.savefig(f'../results/circles{index}.jpg', dpi=1000)
    line_image = show_correspondence_lines(image1, image2,
                                   x1[matches_point[:max_to_show, 0]],
                                   y1[matches_point[:max_to_show, 0]],
                                   x2[matches_point[:max_to_show, 1]],
                                   y2[matches_point[:max_to_show, 1]])
    plt.figure()
    plt.imshow(line_image)
    plt.savefig(f'../results/lines{index}.jpg', dpi=1000)
    print("结果已保存")
















