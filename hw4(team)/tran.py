"""
Notice!
1. 该文件 img_transform.py 实现了作业中的第一、二题，该文件共包含三个类：
    （1） class base_LocalAffine ： 该类是局部仿射变换和图像配准的基本类，其使用反向算法，对目标图像中的每一点计算其在原图像中的位置，
  其所包含的几个函数分别具有以下作用：
        1> function transform ： 该函数为核心函数，其实现具体的坐标映射功能，输入坐标为目标图像的坐标，该函数返回原图像中的坐标；
        2> function interpolation ： 该函数实现二次插值算法，用来计算映射点的灰度值；
        3> function derive_new_picture ： 该函数利用上述两个函数完成目标图像所有像素点灰度值的求取。
    （2） class Pic_local_affine ： 该类实现具体的图像配准算法，即利用局部仿射和反向映射的算法实现人类到猩猩脸部的匹配，该类使用时
  需要给定原图片（人类图片），以及原图片和目标图片的对应控制点（用来计算仿射变换矩阵），其关键函数如下所示：
        1> function affine_fit ： 该函数利用对应的控制点坐标，使用最小二乘法的方程解来计算仿射变换矩阵；
        2> function transform ： 该函数重写 类 base_LocalAffine 中的函数 transform ，功能一样。
    （3） class LocalAffine ： 该类实现了图像的局部仿射变换的算法。主要用逆向变换来完成局部放射变换，这样可以减少图像空洞的出现。该类使用时，
  需要给定原图片，以及原图片需要进行局部仿射的区域，（这个算法基本设定是将图片的四个顶点固定，对标定区域进行局部仿射。）主要由以下4种变换构成：
        1> function scale : 该函数根据参数生成对应的仿射矩阵，然后得到逆变换矩阵，然后利用 derive_new_picture 函数实现逆变换，得到新图片。
        2> function translate : 原理同上。
        3> function rotate : 原理同上。
        4> function shear : 原理同上。

2. 该文件可直接运行，最终能够得到测试局部仿射以及图像配准的结果图片，具体可看 Testing Part 部分。
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class base_LocalAffine:
    def __init__(self, origin_picture_path, new_picture_size=None):
        """
        1. new_picture_size 应该为 人类 的图片；
        """
        
        root_dir, file_name = os.path.split(origin_picture_path)
        pwd = os.getcwd()
        if root_dir:
            os.chdir(root_dir)
        self.img = cv2.imread(file_name, 0).astype(float)
        os.chdir(pwd)
        self.new_picture_size = new_picture_size if new_picture_size is not None else self.img.shape
        self.new_picture = np.zeros(self.new_picture_size)

    def derive_new_picture(self):
        for i in range(self.new_picture_size[0]):
            for j in range(self.new_picture_size[1]):
                nx, ny = self.transform(i, j)
                pixel_val = self.interpolation(nx, ny)
                self.new_picture[i, j] = pixel_val
        return self.new_picture

    def transform(self, x, y):
        """
        1. 注意 该函数为 反向算法，其使用 给定的新图像的坐标 计算得出 原图像对应的坐标点；
        2. 需要注意 计算得出的原图像的坐标点 可能会超出了原图像的范围。
        """
        raise NotImplemented('This function needs to be implemented !')

    def interpolation(self, x, y):
        """
        1. 该函数使用 二次插值 算法，利用所给坐标，计算得出该坐标处的 原图像 的像素值；
        2. 当需要插值的点的坐标超出原图像时，需要考虑如何处理；
        """
        shape = self.img.shape
        if x < 0:
            low, up = 0, 0
        elif x >= shape[0]-1:
            low, up = shape[0]-1, shape[0]-1
        else:
            low, up = int(x), int(x) + 1
        if y < 0:
            left, right = 0, 0
        elif y >= shape[1]-1:
            left, right = shape[1]-1, shape[1]-1
        else:
            left, right = int(y), int(y) + 1
        return ((right - y) * self.img[low, left] + (y - left) * self.img[low, right]) * (up - x) + \
               ((right - y) * self.img[up, left] + (y - left) * self.img[up, right]) * (x - low)


class Pic_local_affine(base_LocalAffine):
    def __init__(self, origin_picture_path, datafrom, datato=None, new_picture_size=None):
        """
        1. 该类实现基于反向映射和局部仿射的图像匹配算法；
        2. 该初始化函数根据所给的控制点（眼睛，鼻子，嘴巴）计算局部仿射需要的参数；
        3. 参数 datafrom 和 datato 均为长度为 M 的列表，其中 M 为局部仿射的个数，每个局部仿射需要至少三个点来控制，因此列表中的是
        N_i*2 的矩阵，其中 N_i 为每个局部仿射输入的控制点。
        """
        super().__init__(origin_picture_path, new_picture_size)
        self.datafrom, self.datato = datafrom, datato
        self.egson = 3

        self.all_local_affine_paras = []
        for i in range(len(datato)):
            used_data = np.c_[datato[i], np.ones([datato[i].shape[0], 1])]
            self.all_local_affine_paras.append(self.affine_fit(used_data, datafrom[i]))

    def affine_fit(self, begindata, afterdata):
        """
        1. 该函数用来求取局部仿射的 六个参数；
        2. 因为采用反向映射，因此参数 begindata 为 人脸 的图像中的坐标，而 afterdata 为 狒狒 的图像中的坐标；
        3. 求取方法使用的是 最小二乘法 的方程解。
        """
        # 计算人脸到狒狒的正向变换 H
        vec1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(begindata.T, begindata)), begindata.T), afterdata[:, 0])
        vec2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(begindata.T, begindata)), begindata.T), afterdata[:, 1])
        H = np.array([vec1.flatten(), vec2.flatten(), np.array([0, 0, 1])])
        # 计算狒狒到人脸的逆变换 T^-1 = (transpose(H)*H)^-1 * transpose(H)
        inv_T = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T)

        return [inv_T[0, :].flatten(), inv_T[1, :].flatten()]

    def transform(self, x, y):
        """
        1. 该函数实现从 猩猩 的图像到 人类 的图像的局部映射；
        2. x, y 应该是猩猩图像中的坐标。
        """
        control_dis = [0 for _ in range(len(self.datafrom))]

        control_weights = []
        all_affine = []

        for i in range(len(self.datafrom)):
            dis = np.mean(np.sqrt(np.sum((self.datafrom[i] - (x, y)) ** 2, axis=1)))
            if dis <= control_dis[i]:
                return np.sum(self.all_local_affine_paras[i][0] * (x, y, 1)), \
                       np.sum(self.all_local_affine_paras[i][1] * (x, y, 1))
            control_weights.append(1 / dis ** self.egson)
            all_affine.append([np.sum(self.all_local_affine_paras[i][0] * (x, y, 1)),
                               np.sum(self.all_local_affine_paras[i][1] * (x, y, 1))])

        sum_weights = sum(control_weights)
        control_weights = [each / sum_weights for each in control_weights]

        nx, ny = 0, 0
        for i in range(len(control_weights)):
            nx += control_weights[i] * all_affine[i][0]
            ny += control_weights[i] * all_affine[i][1]

        return nx, ny

class LocalAffine(Pic_local_affine):
    def __init__(self, origin_picture_path, points, new_picture_size=None):
        """
        1. 该类将图片四个顶点固定，实现某区域的局部仿射
        2. 参数points 是确定局部区域的点
        """
        root_dir, file_name = os.path.split(origin_picture_path)
        pwd = os.getcwd()
        if root_dir:
            os.chdir(root_dir)
        self.img = cv2.imread(file_name, 0).astype(float)
        os.chdir(pwd)
        self.new_picture_size = new_picture_size if new_picture_size is not None else self.img.shape
        self.new_picture = np.zeros(self.new_picture_size)
        self.datafrom = [
            np.array([[0, 0]]),  # 固定图像四个顶点不动
            np.array([[0, self.new_picture_size[1]-1]]),
            np.array([[self.new_picture_size[0]-1, 0]]),
            np.array([[self.new_picture_size[0]-1, self.new_picture_size[1]-1]])
        ]
        self.datafrom.append(points)
        self.egson = 2

        self.all_local_affine_paras = []

    def reset(self):
        self.all_local_affine_paras = [
            [np.array([1, 0, 0]), np.array([0, 1, 0])] for i in range(4)
        ]

    # 以下分别实现局部仿射中的scale rotate translate shear 变换的封装形式
    # 注：每种变换均采用 逆向映射，以防止出现空洞的现象
    def scale(self, size, direction=None):
        """
        实现图像局部仿射的 scale 函数
        :param size: scale 的尺度，如果需要对x，y方向进行不同尺度的scale 则需要输入一个列表
        :param direction:指定方向，默认为x，y两个方向，可以选择“x” 或者 “y” 方向
        """
        self.reset()
        # 处理参数 需要将 size 规范化为列表形式
        if direction is not None:  # 如果指定了方向， size参数必须为一个常数
            assert type(size) == int
            if direction == "x":
                size = [size, 1]
            elif direction == "y":
                size = [1, size]
            else:
                raise ValueError("unknown direction")
        else:
            if type(size) == int:
                size = [size, size]
            else:
                assert type(size) == list and len(size) == 2  # 如果size为列表则需要长度为2

        # 第一步，根据输入参数得到正向的 affine matrix
        H = np.array([[size[0], 0, 0], [0, size[1], 0], [0, 0, 1]])
        # 第二步，得到逆变换矩阵
        inv_T = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T)
        self.all_local_affine_paras += [[inv_T[0, :].flatten(), inv_T[1, :].flatten()]]
        # 第三步，调用函数得到结果
        return self.derive_new_picture()

    def translate(self, x, y):
        """
        现图像局部仿射的 translate 函数
        :param x: x方向移动距离
        :param y: y方向移动距离
        """
        self.reset()
        # 第一步，根据输入参数得到正向的 affine matrix
        H = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
        # 第二步，得到逆变换矩阵
        inv_T = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T)
        self.all_local_affine_paras += [[inv_T[0, :].flatten(), inv_T[1, :].flatten()]]
        # 第三步，调用函数得到结果
        return self.derive_new_picture()

    def rotate(self, angle, clockwise=True):
        """
        现图像局部仿射的 rotate 函数
        :param angle: 旋转的角度 例如30代表30°
        :param clockwise: 默认顺时针
        """
        self.reset()
        theta = np.pi*angle/180 if clockwise else -np.pi*angle/180
        # 第一步，根据输入参数得到正向的 affine matrix
        H = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        # 第二步，得到逆变换矩阵
        inv_T = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T)
        self.all_local_affine_paras += [[inv_T[0, :].flatten(), inv_T[1, :].flatten()]]
        # 第三步，调用函数得到结果
        return self.derive_new_picture()

    def shear(self, angle, direction):
        """
        实现图像局部仿射的 shear 函数
        :param angle: scale 的尺度
        :param direction:指定方向可以选择“x” 或 “y” 方向
        """
        self.reset()
        theta = np.pi * angle / 180
        # 第一步，根据输入参数得到正向的 affine matrix
        if direction == "x":
            H = np.array([[1, np.tan(theta), 0], [0, 1, 0], [0, 0, 1]])
        else:
            H = np.array([[1, 0, 0], [np.tan(theta), 1, 0], [0, 0, 1]])
        # 第二步，得到逆变换矩阵
        inv_T = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T)
        self.all_local_affine_paras += [[inv_T[0, :].flatten(), inv_T[1, :].flatten()]]
        # 第三步，调用函数得到结果
        return self.derive_new_picture()

if __name__ == '__main__':
    # ###########################*** Testing Part ***#############################
    """
    1. 局部仿射 的测试，这部分实例化类 LocalAffine， 并对图像进行相应的局部仿射变换；
    """
    local_pic_path = "./data/local.jpg"  # 测试图片
    # 实现局部仿射的四种变换 scale translate rotate shear
    # 将图像的四个顶点固定，对选定区域进行局部放射
    # loacl_points 控制了需要进行局部放射的区域
    local_points = np.array([
        [108, 108],
        [108, 248],
        [248, 108],
        [248, 248]
    ])
    # 初始化类
    LocalPic = LocalAffine(local_pic_path, local_points)
    # 设置可视化图片大小
    plt.figure(figsize=(20, 20))
    # scale 
    plt.subplot(221)
    local1 = LocalPic.scale([0.5, 1])
    plt.imshow(local1, cmap='gray')
    plt.title("scale 0.5 in x, 1 in y")
    # translate
    plt.subplot(222)
    local2 = LocalPic.translate(50, 50)
    plt.imshow(local2, cmap='gray')
    plt.title("translate 50 in x, 50 in y")
    # rotate
    plt.subplot(223)
    local3 = LocalPic.rotate(30)
    plt.imshow(local3, cmap='gray')
    plt.title("rotate 30°")
    # shear
    plt.subplot(224)
    local4 = LocalPic.shear(30, "y")
    plt.imshow(local4, cmap='gray')
    plt.title("shear 30° in y")
    # 保存并展示
    plt.savefig("./result/first.png")
    plt.show()

    """
    2. 利用 局部仿射 和 反向映射 来实现图片的配准；
    """
    filename = './data/human1.jpg'

    # 实现图像之间的配准，需要提取两张图像的特征，
    # points_rigis 为 猩猩 的相关区域的点的位置；
    # human_points 为 人类 的对应区域的点的位置。
    points_rigid = [np.array([[32, 71],  # 左眼
                            [23, 88],
                            [42, 89],
                            [41, 110]]),
                    np.array([[38, 142],  # 右眼
                            [22, 162],
                            [39, 171],
                            [26, 183]]),
                    np.array([[44, 125],  # 鼻子
                            [194, 76],
                            [192, 167], ]),
                    np.array([[199, 54],  # 嘴巴
                            [244, 98],
                            [246, 159],
                            [198, 201],
                            [229, 122]])]

    human_points = [np.array([[189, 69],  # 左眼
                            [175, 89],
                            [193, 91],
                            [189, 109]]),
                    np.array([[190, 171],  # 右眼
                            [176, 189],
                            [193, 190],
                            [188, 211]]),
                    np.array([[191, 139],  # 鼻子
                            [247, 113],
                            [242, 166]]),
                    np.array([[277, 100],  # 嘴巴
                            [296, 125],
                            [297, 149],
                            [280, 175],
                            [275, 138]])]

    # 初始化类
    pic_aff = Pic_local_affine(filename, points_rigid, human_points, (256, 256))
    # 调用写好的局部仿射及反向映射实现人脸到狒狒脸的转换
    new_pic = pic_aff.derive_new_picture()
    # 可视化
    plt.imshow(new_pic, cmap='gray')
    plt.savefig('./result/second.png')
    plt.show()

