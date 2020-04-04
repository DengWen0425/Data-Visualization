import numpy as np


def board_detection(data, pos, size):
    # this is a function which will return the coordinates of window corner
    x_length, y_length = data.shape
    px, py = pos
    r = int(size / 2)
    ex = px + r + 1
    ey = py + r + 1
    while ex > x_length:
        ex -= 1
    while ey > y_length:
        ey -= 1
    l = size - r - 1
    sx = px - l
    sy = py - l
    while sx < 0:
        sx += 1
    while sy < 0:
        sy += 1
    return sx, ex, sy, ey


def adpt_thrsh_MovAvg(ori_img, window_size, c=1.0):  #  moving average
    """
    run a locally adaptive threshold operation to an image
    :param ori_img: the input image
    :param window_size: The size of the neighborhood
    :param c: hyper parameter for method of moving average
    :return: img after local adaptive threshold
    """

    img = ori_img.copy().astype(int)
    x_len, y_len = img.shape  # the shape of image
    new_img = img.copy()  # copy a new variance to store the new_img

    # initialize -- first window
    cur_left, cur_right, cur_up, cur_down = board_detection(img, (0, 0), window_size)
    neighborhood_num = (cur_right-cur_left)*(cur_down-cur_up)
    neighborhood_sum = np.sum(img[cur_left:cur_right, cur_up:cur_down])
    m = neighborhood_sum * c / neighborhood_num
    new_img[0][0] = 1 if img[0][0] >= m else 0

    # run a loop to update all position
    direction = (0, 1)  # initial direction : right
    i, j = 0, 0
    while True:
        # move the neighborhood
        i, j = i + direction[0], j + direction[1]
        # compute the new boarder
        boarder = board_detection(img, (i, j), window_size)
        # update the neighborhood number
        neighborhood_num = (boarder[1]-boarder[0])*(boarder[3]-boarder[2])
        if direction == (0, 1):  # move right
            add = np.sum(img[boarder[0]:boarder[1], cur_down:boarder[3]])
            delete = np.sum(img[boarder[0]:boarder[1], cur_up:boarder[2]])
            neighborhood_sum = neighborhood_sum + add - delete
        if direction == (1, 0):  # move down
            add = np.sum(img[cur_right:boarder[1], boarder[2]:boarder[3]])
            delete = np.sum(img[cur_left:boarder[0], boarder[2]:boarder[3]])
            neighborhood_sum = neighborhood_sum + add - delete
        if direction == (0, -1):  # move left
            add = np.sum(img[boarder[0]:boarder[1], boarder[2]:cur_up])
            delete = np.sum(img[boarder[0]:boarder[1], boarder[3]:cur_down])
            neighborhood_sum = neighborhood_sum + add - delete
        # update the direction of last step
        cur_left, cur_right, cur_up, cur_down = boarder
        # calculate current position's threshold
        m = neighborhood_sum * c / neighborhood_num
        # update the pixel
        new_img[i][j] = 1 if img[i][j] >= m else 0

        # this is a terminal condition
        if (j == y_len - 1 and i == x_len - 1 and x_len % 2 == 1) or (j == 0 and i == x_len - 1 and x_len % 2 == 0):
            # finish and exit
            break
        # direction judgement to support Z type move
        elif (j == y_len - 1 and i % 2 == 0) or (j == 0 and i % 2 == 1):  # reach to the end of a row of beginning
            direction = (1, 0)
        elif j == y_len - 1 and i % 2 == 1:  # after switching to next row change direction
            direction = (0, -1)
        elif j == 0 and i % 2 == 0:  # after switching to next row change direction
            direction = (0, 1)

    return new_img


def adpt_thrsh_OTSU(img, window_size, c):
    """
    locally adaptive threshold based on OTSU
    :param img:  input image
    :param window_size:  window size
    :param c: hyper parameter
    :return: img
    """
    x_len, y_len = img.shape  # the shape of the img
    new_img = img.copy()  # to store the new img

    # begin of the loop
    for i in range(x_len):
        for j in range(y_len):
            # the boarder of the neighborhood
            boarder = board_detection(img, (i, j), window_size)
            # compute the local threshold
            local_threshold = local_OTSU(img[boarder[0]:boarder[1], boarder[2]:boarder[3]], c)
            # update
            new_img[i][j] = 1 if img[i][j] >= local_threshold else 0

    return new_img


def OTSU_threshold(img):
    """
    Use OTSU algorithm to calculate the threshold of an img
    :param img:  the input image
    :return:  a threshold
    """
    x_len, y_len = img.shape  # the shape of image
    N = x_len * y_len  # the total number of pixel

    # compute the histogram
    hist = np.zeros(256)
    for i in range(x_len):
        for j in range(y_len):
            hist[img[i][j]] += 1

    max_variance = 0  # to keep record of the maximum between-group variance
    threshold = 0  # to keep record of the threshold that maximize the between-group variance

    # run a loop to compute the best threshold
    for t in range(img.min()+1, img.max()):

        n_bk = np.sum(hist[:t])  # the number of pixels in background
        n_fg = np.sum(hist[t:])  # the number of pixels in foreground

        w_bk = n_bk / N  # freq of background
        w_fg = n_fg / N  # freq of foreground
        if n_bk == 0:
            u_bk = 0
        else:
            u_bk = np.sum(hist[:t] * np.arange(t)) / n_bk  # mean intensity of background
        if n_fg == 0:
            u_fg = 0
        else:
            u_fg = np.sum(hist[t:] * np.arange(t, 256)) / n_fg  # mean intensity of foreground

        var_bet = w_bk * w_fg * (u_fg - u_bk) ** 2  # between-group variance

        # to compare with the max_variance
        if var_bet > max_variance:
            max_variance = var_bet  # update variance
            threshold = t  # update threshold

    return threshold


def local_OTSU(img, c):
    """
    run OTSU on every neighborhood is expensive, a better idea is to go over every pixels in the local img
    :param img: input local img
    :param c: hyper parameter
    :return: local threshold
    """
    # initialize
    flat_img = img.flatten().astype(float)  # flatten operation to make later operation more convenience
    # sort the pixels
    flat_img.sort()
    # keep record of the maximum between-class variance
    max_variance = 0
    # keep record of the best threshold
    threshold = 0
    # the number of pixels in the background
    back_total = flat_img[0]
    # the number of pixels in the foreground
    fore_total = np.sum(flat_img[1:])
    # begin a loop
    for i in range(1, flat_img.size):
        # to avoid repeat computation
        if flat_img[i] == flat_img[i-1]:
            back_total += flat_img[i]
            fore_total -= flat_img[i]
            continue
        # mean of background and foreground
        u_bk = back_total/i
        u_fg = fore_total/(flat_img.size-i)
        # freq of background and foreground
        w_bk = i/flat_img.size
        w_fg = 1-w_bk
        # between-class variance
        var_bet = w_fg * w_bk * (u_fg - u_bk) ** 2
        # to compare
        if var_bet > max_variance:
            # update
            threshold = flat_img[i]
            max_variance = var_bet
        # update the the number of pixels in the background and foreground
        back_total += flat_img[i]
        fore_total -= flat_img[i]
    return threshold * c


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    for case in ["test3.1.png", "test3.2.png"]:
        image = np.array(Image.open("./test/" + case).convert("L"))
        threshold = OTSU_threshold(image)
        img2 = image.copy()
        img2[image >= threshold] = 1
        img2[image < threshold] = 0
        plt.clf()
        plt.imshow(img2, cmap=plt.get_cmap('gray'))
        plt.savefig(".result/result" + case[4:8] + "1.png")
        plt.show()
        image3 = adpt_thrsh_MovAvg(image, 4, 0.9)
        plt.clf()
        plt.imshow(image3, cmap=plt.get_cmap('gray'))
        plt.savefig(".result/result" + case[4:8] + "2.png")
        plt.show()
        image4 = adpt_thrsh_OTSU(image, 6, 0.9)
        plt.clf()
        plt.imshow(image4, cmap=plt.get_cmap('gray'))
        plt.savefig(".result/result" + case[4:8] + "3.png")
        plt.show()
    image = np.array(Image.open("./test/test3.3.png").convert("L"))
    threshold = OTSU_threshold(image)
    img2 = image.copy()
    img2[image >= threshold] = 1
    img2[image < threshold] = 0
    plt.clf()
    plt.imshow(img2, cmap=plt.get_cmap('gray'))
    plt.savefig(".result/result3.3.1.png")
    plt.show()
    image3 = adpt_thrsh_MovAvg(image, 7, 0.95)
    plt.clf()
    plt.imshow(image3, cmap=plt.get_cmap('gray'))
    plt.savefig(".result/result3.3.2.png")
    plt.show()
    image4 = adpt_thrsh_OTSU(image, 6, 0.95)
    plt.clf()
    plt.imshow(image4, cmap=plt.get_cmap('gray'))
    plt.savefig(".result/result3.3.3.png")
    plt.show()




