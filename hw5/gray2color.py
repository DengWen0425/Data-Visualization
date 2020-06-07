import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def gray2color(img, colormap="rainbow", save=None):
    """
    该函数用于将 灰度图 向 彩色图(伪彩) 变换。
    算法
    :param save: 是否保存图片
    :param colormap: 用于灰度向彩色映射的分段颜色表，可以调用matplotlib中自带的各种colormap
    :param img: 输入的灰度图图像
    :return:  伪彩图像
    """
    colors = plt.get_cmap(colormap)(range(256))  # 从matplotlib库中获取对应的分段颜色表
    # 由于得到的分段颜色表是 RGBA 的形式，所以要将 RGBA 的形式转换为 RGB 形式
    red = colors[:, 0]
    green = colors[:, 1]
    blue = colors[:, 2]
    # 创建一个空的变量来存储 伪彩 后的图片
    pseudo_colored_img = np.zeros((img.shape[0], img.shape[1], 3))
    # 进行映射
    pseudo_colored_img[:, :, 0] = red[img]
    pseudo_colored_img[:, :, 1] = green[img]
    pseudo_colored_img[:, :, 2] = blue[img]

    # 转换为 uint8 的格式
    pseudo_colored_img = (pseudo_colored_img * 256).astype(np.uint8)

    if save is not None:  # 是否保存
        Image.fromarray(pseudo_colored_img).save("./result/"+save)

    return pseudo_colored_img


# 进行测试
if __name__ == "__main__":
    # 读取测试图片
    image1 = np.array(Image.open("./data/camera.png").convert("L"))
    image2 = np.array(Image.open("./data/cell.png").convert("L"))

    # 测试
    result1 = gray2color(image1, save="camera_rainbow.png")
    result2 = gray2color(image1, colormap="hsv", save="camera_hsv.png")
    result3 = gray2color(image2, save="cell_rainbow.png")
    result4 = gray2color(image2, colormap="hsv", save="cell_hsv.png")



