import numpy as np
from EvaluateHistogram import evaluate_histogram


def board_detection(data, pos, size):
    # this is a function which will return the coordinates of window corner
    x_len, y_len = data.shape
    px, py = pos
    r = int(size / 2)
    ex = px + r + 1
    ey = py + r + 1
    while ex > x_len:
        ex -= 1
    while ey > y_len:
        ey -= 1
    l = size - r - 1
    sx = px - l
    sy = py - l
    while sx < 0:
        sx += 1
    while sy < 0:
        sy += 1
    return sx, ex, sy, ey


def local_histograms(pData, window_size=5, bins=10, intensityMin=0, intensityMax=255, padding=None):
    """
    :param pData: the input picture
    :param window_size: usually odd, if it is even we let the window downward sloping
    :param bins: number of bins
    :param intensityMin:lowest range
    :param intensityMax:highest range
    :param padding: use to fill out-of-board values
    :return: local histograms for every pixel point
    """
    ##
    x_start, y_start = 0, 0
    x_length, y_length = pData.shape
    if padding is not None:
        radius = int(window_size/2)
        pData = np.c_[pData, padding*np.ones((pData.shape[0], radius))]
        pData = np.c_[padding*np.ones((pData.shape[0], window_size-radius-1)), pData]
        x_start += window_size-radius-1
        pData = np.r_[pData, padding*np.ones((radius, pData.shape[1]))]
        pData = np.r_[padding*np.ones((window_size-radius-1, pData.shape[1])), pData]
        y_start += window_size-radius-1
    histograms = {}
    cur_hist = None
    for x in range(x_start, x_start+x_length):
        for y in range(y_start, y_start+y_length):
            if cur_hist is None:  # initialize where there is not a histogram
                # will call a evaluate_histogram directly
                cur_left, cur_right, cur_up, cur_down = board_detection(pData, (x, y), window_size)
                local_pic = pData[cur_left:cur_right, cur_up:cur_down].reshape((1, -1))
                cur_hist = evaluate_histogram(local_pic, local_pic.size, 1, bins, intensityMin, intensityMax)
            else:  # there is already a histogram to use local update
                cur_left, cur_right, cur_up, cur_down = board_detection(pData, (x, y), window_size)

                if cur_left > old_left:
                    # do deletion
                    delete_pic = pData[old_left:cur_left, old_up:old_down].reshape((1, -1))
                    cur_hist = cur_hist - evaluate_histogram(delete_pic, delete_pic.size, 1, bins, intensityMin,
                                                             intensityMax)
                if cur_right > old_right:
                    # add to hist
                    add_pic = pData[old_right:cur_right, old_up:old_down].reshape((1, -1))
                    cur_hist = cur_hist + evaluate_histogram(add_pic, add_pic.size, 1, bins, intensityMin,
                                                             intensityMax)

                if cur_up > old_up:
                    # do deletion
                    delete_pic = pData[old_left:old_right, old_up:cur_up].reshape((1, -1))
                    cur_hist = cur_hist - evaluate_histogram(delete_pic, delete_pic.size, 1, bins, intensityMin,
                                                             intensityMax)

                if cur_down > old_down:
                    # add to hist
                    add_pic = pData[old_left:old_right, old_down:cur_down].reshape((1, -1))
                    cur_hist = cur_hist + evaluate_histogram(add_pic, add_pic.size, 1, bins, intensityMin,
                                                             intensityMax)

            histograms[(x, y)] = cur_hist
            old_left, old_right, old_up, old_down = cur_left, cur_right, cur_up, cur_down
        cur_hist = histograms[(x, 0)]
        old_left, old_right, old_up, old_down = board_detection(pData, (x, 0), window_size)
    return histograms


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(123)
    data = np.c_[np.random.normal(0.5, 0.04, (50, 25)), np.random.normal(1.5, 0.04, (50, 25))]
    plt.subplot(221)
    plt.imshow(data, cmap=plt.get_cmap('gray'))
    plt.title("(a).", y=-0.3)
    local_hists = local_histograms(data, 5, 5, 0, 2, None)
    plt.subplot(222)
    plt.imshow(np.array([local_hists[(3, 3)]]), cmap=plt.get_cmap('gray'))
    plt.xticks(range(5), [i / 2 for i in range(5)])
    plt.title("(b).", y=-0.9)
    plt.subplot(223)
    plt.imshow(np.array([local_hists[(40, 40)]]), cmap=plt.get_cmap('gray'))
    plt.xticks(range(5), [i / 2 for i in range(5)])
    plt.title("(c).", y=-0.9)
    plt.subplot(224)
    plt.imshow(np.array([local_hists[(24, 24)]]), cmap=plt.get_cmap('gray'))
    plt.xticks(range(5), [i / 2 for i in range(5)])
    plt.title("(d).", y=-0.9)
    print(local_hists[(3, 3)])
    print(local_hists[(40, 40)])
    print(local_hists[(24, 24)])
    plt.savefig("problem1-3.png")
    plt.show()




