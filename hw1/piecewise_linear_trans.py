import numpy as np


def piecewise_linear_transform(pData, L):
    """
    :param pData: input pic should be the form of np.array
    :param L: the transformation range parameter
    :return: a new pic after transformation
    """
    def single_op(p):
        if 0 <= p < L*3/8:
            return p/3
        elif L*3/8 <= p < L*5/8:
            return 3*p-L
        elif L*5/8 <= p <= L-1:
            return (p+2*L)/3

    newData = pData.copy()
    x_length, y_length = pData.shape
    for x in range(x_length):
        for y in range(y_length):
            newData[x, y] = single_op(pData[x, y])

    return newData


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = np.array([np.arange(96, 156) for _ in range(256)])
    data = np.c_[data, np.array([np.arange(156, 256) for _ in range(256)])]
    data = np.c_[np.array([np.arange(0, 96) for _ in range(256)]), data]
    print(data)
    plt.subplot(211)
    plt.imshow(data, cmap=plt.get_cmap('gray'))
    plt.title("(a).", y=-0.3)
    new_data = piecewise_linear_transform(data, 256)
    plt.subplot(212)
    plt.imshow(new_data, cmap=plt.get_cmap('gray'))
    plt.title("(b).", y=-0.3)
    print(new_data)
    plt.savefig("problem2.png")
    plt.show()



