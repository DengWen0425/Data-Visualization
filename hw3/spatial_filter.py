# This is for problem 1. which aims to implement smoothing operation and sharpening operation with spatial filters

import numpy as np


def gaussian_smoothing(ori_img, kernel_size, sigma, padding="replicate"):
    """
    smoothing operation based on weighted average using a gaussian kernel
    :param ori_img:  input image
    :param kernel_size: the size of the kernel
    :param sigma: the std of the gaussian kernel, if sigma = 0, it is a box kernel
    :param padding: the way of padding
    :return: the image after smoothing
    """
    img = ori_img.copy()
    kernel = np.ones((kernel_size, kernel_size), np.float)  # initialize a kernel
    center = kernel_size // 2  # the center position of the kernel

    # initialize a 2d gaussian kernel based on the sigma
    if sigma != 0:
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(-0.5*((i-center)**2+(j-center)**2)/sigma**2)  # gaussian probability

    kernel = kernel/kernel.sum()  # normalize

    width, height = img.shape  # the width and height of an image

    # padding
    if padding == "zero":  # padding with number zero
        new_image = np.zeros((width+center*2, height+center*2), np.float)
        new_image[center:(center+width), center:(center+height)] = img

    elif padding == "replicate":  # padding by fill with the nearest pixel value
        new_image = np.c_[img, np.array([img[:, -1] for i in range(center)]).T]
        new_image = np.c_[np.array([new_image[:, 1] for i in range(center)]).T, new_image]
        new_image = np.r_[new_image, np.array([new_image[-1, :] for i in range(center)])]
        new_image = np.r_[np.array([new_image[0, :] for i in range(center)]), new_image]

    elif padding == "mirror":  # padding by mirror-reflecting the image across its border
        new_image = np.c_[img.copy(), img[:, :kernel_size].copy()]
        new_image = np.c_[img[:, -kernel_size:].copy(), new_image]
        tmp = new_image[-kernel_size:, :].copy()
        new_image = np.r_[new_image, new_image[:kernel_size, :].copy()]
        new_image = np.r_[tmp, new_image]

    # begin a loop to deal with every pixels
    for i in range(center, center+width):
        for j in range(center, center+height):
            # calculate the boarder of the neighborhood
            left, right = i-center, i+kernel_size-center
            up, down = j-center, j+kernel_size-center
            # convolution
            img[i-center, j-center] = np.sum(new_image[left:right, up:down] * kernel)

    return img


def median_smoothing(ori_img, kernel_size):
    """
    smoothing operation based on nonlinear order-statistic smoothing -- median
    :param ori_img: input image
    :param kernel_size:  the size of the kernel
    :return:  the image after smoothing
    """
    img = ori_img.astype(np.float).copy()
    radius = kernel_size//2  # the radius of a kernel

    width, height = img.shape  # the width and height of an image

    new_img = np.zeros(img.shape)  # create a new variable to store the new image

    # begin a loop to deal every pixel
    for i in range(width):
        for j in range(height):
            # calculate the boarder of the neighborhood
            left, right = max(0, i-radius), min(width, i+kernel_size-radius)
            up, down = max(0, j-radius), min(height, j+kernel_size-radius)
            # calculate the median
            new_img[i, j] = np.median(img[left:right, up:down])

    return new_img


# pre-defined kernels from the ppt
kernels = [
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float),
    np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float)
]


def laplacian_sharpening(ori_img, kernel, weight):
    """
    sharpening operation based on laplacian operator
    :param ori_img:  input image
    :param kernel:  the kind of laplacian kernels we used, can be choose form kernels. Can be 0 or 1.
    :param weight: the weight of the Laplacian Filter image
    :return: image after sharpening
    """
    img = ori_img.astype(np.float).copy()
    kernel = kernels[kernel]  # choose a kernel

    width, height = img.shape  # the width and height of an image

    laplacian_image = img.copy()  # to store the new image

    # begin a loop to deal every pixel
    for i in range(1, width-1):
        for j in range(1, height-1):
            # fetch the neighborhood
            tmp = img[i-1:i+2, j-1:j+2]
            # calculate the 2-rd derivative
            laplacian_image[i, j] = np.sum(tmp*kernel)

    new_image = img - weight*laplacian_image  # result need to minus the Laplacian filter image
    new_image[new_image < 0] = 0

    return new_image, laplacian_image


def highboost_sharpening(ori_img, weight, kernel_size, sigma):
    """
    sharpening operation based on highboost method, the blur image is got form gaussian_smoothing
    :param ori_img: the input image
    :param weight: the weight of the mask image
    :param kernel_size: the kernel size to get the blur image
    :param sigma: the std of the gaussian kernel
    :return: image after sharpening, blur image, mask image
    """
    img = ori_img.copy().astype(np.float)
    blur_img = gaussian_smoothing(img, kernel_size, sigma)  # to get the blur image
    mask_img = img - blur_img
    # mask_img = (mask_img - mask_img.min()) / (mask_img.max()-mask_img.min())
    # mask_img *= 255
    new_img = img + weight*mask_img  # highboost formula
    return new_img, blur_img, mask_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image

    # test for smoothing
    image1 = np.array(Image.open("./test_data/problem1.png").convert("L"))

    # box filter
    box1 = gaussian_smoothing(image1, 7, 0)
    box2 = gaussian_smoothing(image1, 21, 0)

    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image1, cmap=plt.get_cmap('gray'))
    plt.title("origin")
    plt.subplot(132)
    plt.imshow(box1, cmap=plt.get_cmap('gray'))
    plt.title("5x5")
    plt.subplot(133)
    plt.imshow(box2, cmap=plt.get_cmap('gray'))
    plt.title("15x15")
    plt.savefig("./results/result1.1.png")
    plt.show()

    # gaussian filter
    gauss1 = gaussian_smoothing(image1, 13, 2)
    gauss2 = gaussian_smoothing(image1, 21, 3.5)
    gauss3 = gaussian_smoothing(image1, 43, 7)

    plt.clf()
    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.imshow(image1, cmap=plt.get_cmap('gray'))
    plt.title("origin")
    plt.subplot(222)
    plt.imshow(gauss1, cmap=plt.get_cmap('gray'))
    plt.title("13x13,sigma=2")
    plt.subplot(223)
    plt.imshow(gauss2, cmap=plt.get_cmap('gray'))
    plt.title("21x21,sigma=3.5")
    plt.subplot(224)
    plt.imshow(gauss3, cmap=plt.get_cmap('gray'))
    plt.title("43x43, sigma=7")
    plt.savefig("./results/result1.2.png")
    plt.show()

    # padding difference
    zero_p = gaussian_smoothing(image1, 121, 20, "zero")
    replicate_p = gaussian_smoothing(image1, 121, 20, "replicate")
    mirror_p = gaussian_smoothing(image1, 121, 20, "mirror")

    plt.clf()
    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.imshow(image1, cmap=plt.get_cmap('gray'))
    plt.title("origin")
    plt.subplot(222)
    plt.imshow(zero_p, cmap=plt.get_cmap('gray'))
    plt.title("zero")
    plt.subplot(223)
    plt.imshow(replicate_p, cmap=plt.get_cmap('gray'))
    plt.title("replicate")
    plt.subplot(224)
    plt.imshow(mirror_p, cmap=plt.get_cmap('gray'))
    plt.title("mirror")
    plt.savefig("./results/result1.3.png")
    plt.show()

    # median filter
    image2 = np.array(Image.open("./test_data/problem1.2.png").convert("L"))
    gauss_p = gaussian_smoothing(image2, 9, 1.5)
    median_p = median_smoothing(image2, 8)

    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image2, cmap=plt.get_cmap('gray'))
    plt.title("origin")
    plt.subplot(132)
    plt.imshow(gauss_p, cmap=plt.get_cmap('gray'))
    plt.title("gaussian")
    plt.subplot(133)
    plt.imshow(median_p, cmap=plt.get_cmap('gray'))
    plt.title("median")
    plt.savefig("./results/result1.4.png")
    plt.show()

    # sharpen

    # Laplacian
    image3 = np.array(Image.open("./test_data/problem1.3.png").convert("L"))
    sharp1, laplacian1 = laplacian_sharpening(image3, 0, 1)
    sharp2, _ = laplacian_sharpening(image3, 1, 1)

    plt.clf()
    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.imshow(image3, cmap=plt.get_cmap('gray'))
    plt.title("origin")
    plt.subplot(222)
    plt.imshow(laplacian1, cmap=plt.get_cmap('gray'))
    plt.title("laplacian filter kernel=1, w=1")
    plt.subplot(223)
    plt.imshow(sharp1, cmap=plt.get_cmap('gray'))
    plt.title("kernel=1, w=1")
    plt.subplot(224)
    plt.imshow(sharp2, cmap=plt.get_cmap('gray'))
    plt.title("kernel=2, w=1")
    plt.savefig("./results/result1.5.png")
    plt.show()

    # highboost
    image4 = np.array(Image.open("./test_data/problem1.4.png").convert("L"))
    high1, blur1, mask1 = highboost_sharpening(image4, 1, 5, 1)
    high2, _, __ = highboost_sharpening(image4, 4.5, 5, 1)

    plt.clf()
    plt.figure(figsize=(30, 20))
    plt.subplot(231)
    plt.imshow(image4, cmap=plt.get_cmap('gray'))
    plt.title("origin")
    plt.subplot(232)
    plt.imshow(blur1, cmap=plt.get_cmap('gray'))
    plt.title("blur")
    plt.subplot(233)
    plt.imshow(mask1, cmap=plt.get_cmap('gray'))
    plt.title("mask")
    plt.subplot(234)
    plt.imshow(high1, cmap=plt.get_cmap('gray'))
    plt.title("highboost k=1")
    plt.subplot(235)
    plt.imshow(high2, cmap=plt.get_cmap('gray'))
    plt.title("highboost k=4.5")
    plt.savefig("./results/result1.6.png")
    plt.show()























