import numpy as np


def basic_global_thrsh(img):
    """
    compute the threshold based on bgt algorithm
    :param img:  input image
    :return:  threshold
    """
    x_len, y_len = img.shape  # the shape of image
    N = x_len * y_len  # the total number of pixel

    # compute the histogram
    pixel_sum = 0
    hist = np.zeros(256)
    for i in range(x_len):
        for j in range(y_len):
            pixel_sum += img[i][j]
            hist[img[i][j]] += 1

    # initial the threshold equal to mean intensity
    old_t = pixel_sum/N

    while True:
        # modify
        t = int(old_t) + 1

        n_bk = np.sum(hist[:t])  # the number of pixels in background
        n_fg = np.sum(hist[t:])  # the number of pixels in foreground

        u_bk = np.sum(hist[:t] * np.arange(t)) / n_bk  # mean intensity of background
        u_fg = np.sum(hist[t:] * np.arange(t, 256)) / n_fg  # mean intensity of foreground

        # update
        new_t = (u_bk + u_fg) / 2

        # terminal condition
        if abs(new_t - old_t) < (img.max() - img.min())/1000:
            break

        old_t = new_t

    return old_t


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    image = np.array(Image.open("./test/test1.png").convert("L"))
    plt.subplot(121)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    threshold = basic_global_thrsh(image)
    image2 = image.copy()
    image2[image >= threshold] = 1
    image2[image < threshold] = 0
    plt.subplot(122)
    plt.imshow(image2, cmap=plt.get_cmap('gray'))
    plt.savefig("./result/result1.png")
    plt.show()



