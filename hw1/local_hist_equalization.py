import numpy as np
from EvaluateHistogram import evaluate_histogram
from local_update_hist import local_histograms


def histogram_equalization(pData, bins=256, intensityMin=0, intensityMax=255):
    flatten_pic = pData.reshape((1, -1))
    histogram = evaluate_histogram(flatten_pic, flatten_pic.size, 1, bins, intensityMin, intensityMax)
    cdf = histogram.cumsum()
    cdf = cdf*255/cdf[-1]
    new_pic = np.interp(flatten_pic, np.linspace(intensityMin, intensityMax, bins), cdf)
    return new_pic.reshape(pData.shape)


def efficient_local_hist_eq(pData, window_size=5, bins=256, intensityMin=0, intensityMax=255, padding=None):
    local_hists = local_histograms(pData, window_size, bins, intensityMin, intensityMax, padding)
    new_pic = np.zeros(pData.shape)
    for pos in local_hists.keys():
        x, y = pos
        hist = local_hists[pos]
        cdf = hist.cumsum()
        cdf = cdf*255/cdf[-1]
        new_pic[x, y] = np.interp(pData[x, y], np.linspace(intensityMin, intensityMax, bins), cdf)

    return new_pic


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    img = np.array(Image.open("problem3.png").convert("L"))
    img1 = histogram_equalization(img)
    plt.subplot(211)
    plt.imshow(img1, cmap=plt.get_cmap('gray'))
    plt.title("(a).", y=-0.3)
    img2 = efficient_local_hist_eq(img)
    plt.subplot(212)
    plt.imshow(img2, cmap=plt.get_cmap('gray'))
    plt.title("(b).", y=-0.3)
    plt.savefig("problem3-1.png")
    plt.show()




