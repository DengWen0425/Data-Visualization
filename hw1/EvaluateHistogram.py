import numpy as np
from functools import reduce


def evaluate_histogram(pData, nData, dimension, bins, intensityMin, intensityMax):
    """
    :param pData: Input data
    :param nData: Number of data
    :param dimension: dimensions of data
    :param bins: number of bins
    :param intensityMin: lowest range
    :param intensityMax: highest range
    :return: return a histogram
    """

    # parameters normalization in which we allow users to only input an integer for some parameters for convenience

    # bins
    try:
        n = len(bins)
        if n != dimension:
            raise ValueError('the dimension of bins must be equal to the dimension of the data')
    except TypeError:
        # bins has only one dimension in which means all the dimensions has the same bins
        bins = dimension * [bins]
    # intensity
    try:
        n = len(intensityMin)
        if n != dimension:
            raise ValueError('the dimension of intensities must  equal to dimensions of data')
    except TypeError:
        # intensity has only one dimension in which means all the dimensions has the same intensity range
        intensityMin = dimension * [intensityMin]
    try:
        m = len(intensityMax)
        if m != dimension:
            raise ValueError('the dimension of intensities must  equal to dimensions of data')
    except TypeError:
        # intensity has only one dimension in which means all the dimensions has the same intensity range
        intensityMax = dimension * [intensityMax]

    bin_space = [0 for i in range(dimension)]
    bin_pos = [0 for i in range(dimension)]
    p_histogram = np.zeros(reduce(lambda x, y: x*y, bins), dtype=int)

    for i in range(dimension):
        if bins[i] <= 0:
            raise ValueError("bins must be positive!")
        bin_space[i] = (intensityMax[i] - intensityMin[i])/bins[i]

    for i in range(nData):
        for j in range(dimension):
            value = pData[j][i]
            bin_pos[j] = int((value - intensityMin[j])/bin_space[j])
            # out-of-boundary detection
            bin_pos[j] = max(bin_pos[j], 0)
            bin_pos[j] = min(bin_pos[j], bins[j]-1)

        index = bin_pos[0]
        for dim in range(1, dimension):
            size = 1
            for idv in range(dim):
                size *= bins[idv]
            index += size*bin_pos[dim]
        p_histogram[index] += 1

    return p_histogram.reshape(bins)


# run a test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # randomly generate 1000 two-dimensional sample
    np.random.seed(123)
    data = np.random.normal(0.5, 0.01, (2, 1000))
    histogram = evaluate_histogram(data, 1000, 2, 10, 0, 1)
    pic = plt.imshow(histogram, cmap=plt.get_cmap('gray'))
    plt.yticks(range(10), [i/10 for i in range(11)])
    plt.xticks(range(10), [i / 10 for i in range(11)])
    plt.colorbar(pic)
    plt.savefig("problem1-1.png")
    plt.show()





