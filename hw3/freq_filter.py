# this is to implement frequency filter to do low-pass smoothing and frequency operation

import numpy as np


def ideal_filter(p_shape, d0, low_pass=True, bias=0):
    """
    to generate an Ideal low-pass Filter based on given shape and threshold
    :param p_shape:  the shape of the filter
    :param d0:  the threshold of the filter
    :param low_pass: high-pass or low-pass, default True, means low-pass
    :param bias: avoiding the the disappearance of Fc
    :return: an ideal filter
    """
    width, height = p_shape  # width and height
    c_x, c_y = width//2, height//2  # the center point
    ideal = np.zeros(p_shape, np.float)  # initialize a filter
    for i in range(width):
        for j in range(height):
            d = np.sqrt((i-c_x)**2 + (j-c_y)**2)  # calculate D(u, v)
            ideal[i, j] = d < d0

    if not low_pass:  # the type of the filter
        ideal = 1-ideal

    ideal = ideal + bias  # add the bias parameter
    return ideal


def gauss_filter(p_shape, d0, low_pass=True, bias=0):
    """
    to generate a gauss low-pass Filter based on given shape and threshold
    :param p_shape:  the shape of the filter
    :param d0:  the std of the gaussian distribution
    :param low_pass: high-pass or low-pass, default True, means low-pass
    :param bias: avoiding the the disappearance of Fc
    :return: an ideal filter
    """
    width, height = p_shape  # width and height
    c_x, c_y = width//2, height//2  # the center point
    gauss_f = np.zeros(p_shape, np.float)  # initialize a filter
    for i in range(width):
        for j in range(height):
            d = (i-c_x)**2 + (j-c_y)**2  # calculate D(u, v)
            gauss_f[i, j] = np.exp(-0.5*d/(d0**2))  # gaussian H(u, v)

    if not low_pass:  # the type of the filter
        gauss_f = 1 - gauss_f

    gauss_f = gauss_f + bias  # add the bias parameter
    return gauss_f


def freq_filter(img, method, d0, low_pass=True, bias=0):
    """
    realize frequency filter operation
    :param img:  the input image
    :param method:  the way, Ideal or Gaussian
    :param d0:  When Ideal: its the threshold of the filter, When gaussian: its the std of gauss dist
    :param low_pass:  the type of the filter
    :param bias:  avoiding the the disappearance of Fc
    :return:  image after filtering
    """
    width, height = img.shape  # the width and height of image
    new_image = np.zeros((width*2, height*2), np.float)  # avoid alias to generate P * Q image
    new_image[0:width, 0:height] = img.copy()  # copy values

    # get the filter
    if method == "ideal":
        h_filter = ideal_filter(new_image.shape, d0, low_pass, bias)
    else:
        h_filter = gauss_filter(new_image.shape, d0, low_pass, bias)

    # multiply (-1)^(x+y) to shift the low-freq to center
    for i in range(width*2):
        for j in range(height*2):
            new_image[i, j] *= (-1)**(i+j)

    # Fourier transform
    f_img = np.fft.fft2(new_image)

    # filtering
    G_img = f_img * h_filter
    # inverse convert
    g_img = np.fft.ifft2(G_img)
    g_img = np.real(g_img)
    # inverse shift op
    for i in range(width*2):
        for j in range(height*2):
            g_img[i, j] *= (-1)**(i+j)

    # fetch result
    result = g_img[:width, :height]

    return result, np.log(np.abs(f_img)+1), np.log(np.abs(G_img)+1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image

    # Ideal filter
    image1 = np.array(Image.open("./test_data/problem3.1.png").convert("L"))
    ideal1, f_ideal1, G_ideal1 = freq_filter(image1, "ideal", 50)
    ideal2, f_ideal2, G_ideal2 = freq_filter(image1, "ideal", 200)

    plt.clf()
    plt.figure(figsize=(30, 20))
    plt.subplot(231)
    plt.imshow(f_ideal1, cmap=plt.get_cmap('gray'))
    plt.title("a")
    plt.subplot(232)
    plt.imshow(G_ideal1, cmap=plt.get_cmap('gray'))
    plt.title("b")
    plt.subplot(233)
    plt.imshow(ideal1, cmap=plt.get_cmap('gray'))
    plt.title("c")
    plt.subplot(234)
    plt.imshow(f_ideal2, cmap=plt.get_cmap('gray'))
    plt.title("d")
    plt.subplot(235)
    plt.imshow(G_ideal2, cmap=plt.get_cmap('gray'))
    plt.title("e")
    plt.subplot(236)
    plt.imshow(ideal2, cmap=plt.get_cmap('gray'))
    plt.title("f")
    plt.savefig("./results/result3.1.png")
    plt.show()

    # Gauss filter
    gauss1, f_gauss1, G_gauss1 = freq_filter(image1, "gauss", 30)
    gauss2, f_gauss2, G_gauss2 = freq_filter(image1, "gauss", 100)

    plt.clf()
    plt.figure(figsize=(30, 20))
    plt.subplot(231)
    plt.imshow(f_gauss1, cmap=plt.get_cmap('gray'))
    plt.title("a")
    plt.subplot(232)
    plt.imshow(G_gauss1, cmap=plt.get_cmap('gray'))
    plt.title("b")
    plt.subplot(233)
    plt.imshow(gauss1, cmap=plt.get_cmap('gray'))
    plt.title("c")
    plt.subplot(234)
    plt.imshow(f_gauss2, cmap=plt.get_cmap('gray'))
    plt.title("d")
    plt.subplot(235)
    plt.imshow(G_gauss2, cmap=plt.get_cmap('gray'))
    plt.title("e")
    plt.subplot(236)
    plt.imshow(gauss2, cmap=plt.get_cmap('gray'))
    plt.title("f")
    plt.savefig("./results/result3.2.png")
    plt.show()

    # frequency operation
    test_img = np.array(Image.open("./test_data/freq_testimage_shepplogan.PNG").convert("L"))
    width, height = test_img.shape  # the width and height of image
    new_img = np.zeros((width * 2, height * 2), np.float)  # avoid alias to generate P * Q image
    new_img[0:width, 0:height] = test_img.copy()  # copy values
    # multiply (-1)^(x+y) to shift the low-freq to center
    for i in range(width * 2):
        for j in range(height * 2):
            new_img[i, j] *= (-1) ** (i + j)

    # Fourier transform
    f_image = np.fft.fft2(new_img)
    # visualize the spectrum
    f_figure = np.log(np.abs(f_image)+1)
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(f_figure, cmap=plt.get_cmap('gray'))
    plt.savefig("./results/result3.3.png")
    plt.show()
    # standardize the image
    f_figure = (f_figure-f_figure.min())/(f_figure.max()-f_figure.min())
    f_figure = f_figure*255
    # save the spectrum
    f_save = Image.fromarray(f_figure.astype(np.uint8))
    f_save.save("./results/freq_image.png")

    # read the points data we saved
    white_points = []
    with open("./points.txt") as p:
        raw = p.readlines()
        for line in raw:
            line = line.strip().split("\t")
            white_points.append((int(float(line[0])), int(float(line[1]))))

    # create a filter
    def gaussian_point(shape, center, d0):
        result = np.zeros(shape)
        r = 70
        left, right = max(center[0]-r, 0), min(shape[0], center[0]+r)
        up, down = max(center[1] - r, 0), min(shape[1], center[1] + r)
        for i in range(left, right):
            for j in range(up, down):
                d = (i - center[0]) ** 2 + (j - center[1]) ** 2  # calculate D(u, v)
                if np.sqrt(d) > 70:
                    continue
                result[i, j] = np.exp(-0.5 * d / (d0 ** 2))  # gaussian H(u, v)
        return result

    _filter = np.zeros(f_image.shape)
    for point in white_points:
        x, y = point  # white point
        _x, _y = 2*width-x, 2*height-y  # the point of symmetry
        _filter += (gaussian_point(_filter.shape, (x, y), 75) + gaussian_point(_filter.shape, (_x, _y), 75))

    _filter = 1-_filter

    # visualize filter
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.imshow(_filter, cmap=plt.get_cmap('gray'))
    plt.savefig("./results/result3.5.png")
    plt.show()

    # filtering
    G_img = _filter * f_image

    # plot the filtering result
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.imshow(np.log(np.abs(G_img)+1), cmap=plt.get_cmap('gray'))
    plt.savefig("./results/result3.4.png")
    plt.show()

    # inverse convert
    g_img = np.fft.ifft2(G_img)
    g_img = np.real(g_img)
    for i in range(g_img.shape[0]):
        for j in range(g_img.shape[1]):
            g_img[i, j] *= (-1) ** (i + j)

    g_img = g_img[:width, :height]

    # save the result
    g_img = (g_img-g_img.min())/(g_img.max()-g_img.min())
    g_img = g_img*255

    f_save = Image.fromarray(g_img.astype(np.uint8))
    f_save.save("./results/3.5result_image.png")

    # Another ways
    for k in ["1", "2"]:
        _filter1 = np.array(Image.open("./results/filter" + k + ".png").convert("L").resize((1432, 1430)))
        _filter1[_filter1 != 0] = 1

        # visualize
        plt.clf()
        plt.figure(figsize=(15, 15))
        plt.imshow(_filter1, cmap=plt.get_cmap('gray'))
        plt.savefig("./results/result3_fmore"+k+".png")
        plt.show()

        G_img1 = _filter1 * f_image

        # plot the filtering result
        plt.clf()
        plt.figure(figsize=(15, 15))
        plt.imshow(np.log(np.abs(G_img1) + 1), cmap=plt.get_cmap('gray'))
        plt.savefig("./results/result3_gmore" + k + ".png")
        plt.show()

        g_img1 = np.fft.ifft2(G_img1)
        g_img1 = np.real(g_img1)
        for i in range(g_img1.shape[0]):
            for j in range(g_img1.shape[1]):
                g_img1[i, j] *= (-1) ** (i + j)
        g_img1 = g_img1[:width, :height]

        g_img1 = (g_img1 - g_img1.min()) / (g_img1.max() - g_img1.min())
        g_img1 = g_img1 * 255

        f_save = Image.fromarray(g_img1.astype(np.uint8))
        f_save.save("./results/3.5result_image" + k + ".png")






























