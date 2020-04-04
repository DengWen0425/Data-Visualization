import numpy as np


def linear_interp(img, n):
    """
    resize an image to enlarge resolution by n times
    :param img:
    :param n: the number of the enlarge times
    :return:
    """
    # this function is used to compute the the pixel value in the new image
    def single_interp(pdata, point):
        """
        linear Interpolation  for a single point
        :param pdata: Given img data
        :param point: Given point which need to compute
        :return: the result of Interpolation
        """
        x, y = point
        x -= 1
        y -= 1
        try:
            return pdata[x][y]
        except IndexError:
            x1, y1 = int(x), int(y)
            x2, y2 = x1 + 1, y1 + 1
            result = pdata[x1][y1] * (x2 - x) * (y2 - y) + pdata[x2][y1] * (x - x1) * (y2 - y) + pdata[x1][y2] * (
                        x2 - x) * (y - y1) + pdata[x2][y2] * (x - x1) * (y - y1)  # based on the linear interp formula
            return int(result)
    # the shape of original image
    x_len, y_len = img.shape
    # the shape of new image
    new_width, new_height = x_len*n, y_len*n
    # define a variable to store new image
    new_img = np.zeros((new_width, new_height))
    # begin a loop
    for i in range(new_width):
        for j in range(new_height):
            # i, j corresponding to the i/n, j/n point in original image
            point = (i/n, j/n)
            # use the function pre-defined to compute the pixel value
            new_img[i][j] = single_interp(img, point)

    return new_img


if __name__ == "__main__":
    from PIL import Image
    for case in ["./test/test4.1.png", "./test/test4.2.png"]:
        image = np.array(Image.open(case).convert("L"))
        image2 = linear_interp(image, 4)
        image2 = Image.fromarray(image2).convert("RGB")
        image2.save(".result/result4." + case[6] + ".png")







