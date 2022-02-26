import numpy as np


# 学习并且引用自lab2TA demonstration_01.ipynb conv_2d!!!
def conv2d(img, krnl):
    krnl_h, krnl_w = krnl.shape[:2]
    pad_h = (krnl_h - 1) // 2
    pad_w = (krnl_w - 1) // 2
    img = np.pad(img, [(pad_h, pad_h), (pad_w, pad_w)], 'constant')
    img_h, img_w = img.shape[:2]
    shape = (img_h - krnl_h + 1, img_w - krnl_w + 1, krnl_h, krnl_w)
    strides = np.array([img_w, 1, img_w, 1]) * img.itemsize
    img = np.lib.stride_tricks.as_strided(img, shape, strides)
    return np.tensordot(img, krnl, axes=[(2, 3), (0, 1)])


def my_imfilter(image, filter):
    """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    temp_img = np.zeros_like(image)
    # 通过zeros_like函数先构建一个和Image相同大小的全0np数组

    filter_image = temp_img

    # 如果image有第三个坐标元素，则固定0，1坐标，遍历2坐标从0到shape[2],若没有则直接conv2d
    if len(image.shape) != 2:
        for i in range(image.shape[2]):
            filter_image[:, :, i] = conv2d(image[:, :, i], filter)
    else:
        filter_image = conv2d(image, filter)

    return filter_image


def create_hybrid_image(image1, image2, filter):
    """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    low_frequencies = my_imfilter(image1, filter)
    # 使用自己写的my_imfilter处理出低频图像
    high_frequencies = image2 - my_imfilter(image2, filter)
    # 根据Hint使用原图像减去低频图像
    hybrid_image = low_frequencies + high_frequencies
    # 将Image1的低频结合Image2的高频

    high_frequencies = high_frequencies + 1 - high_frequencies.max()

    low_frequencies = np.clip(low_frequencies, 0, 1)
    high_frequencies = np.clip(high_frequencies, 0, 1)
    hybrid_image = np.clip(hybrid_image, 0, 1)

    return low_frequencies, high_frequencies, hybrid_image
