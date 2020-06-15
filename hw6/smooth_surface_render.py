import nibabel as nib
import numpy as np
import vtk

# 1.---------------------reader 过程
"""  #---一开始使用 nibabel 来加载 .nii 文件 , 后来在 vtk 官网找到了 vtkNIFTIImageReader 类。。。。

image_lr = nib.load('./data/image_lr.nii.gz')  # 使用 nibabel 读取数据
np_data = np.asanyarray(image_lr.dataobj)  # 转成 numpy 的格式
dims = np_data.shape  # 三个维度的大小
# pixdim 中储存的信息是每个维度网格的间距 grid spacing ， unit per dimension
# more info in https://brainder.org/2012/09/23/the-nifti-file-format/
grid_spacing = image_lr.header["pixdim"][1:4]

# 初始化一个对象 用于储存 上面读入的数据
# more info in https://vtk.org/doc/nightly/html/classvtkImageData.html
image = vtk.vtkImageData()
# 设置维度
image.SetDimensions(dims[0], dims[1], dims[2])
# 设置网格间隔
image.SetSpacing(grid_spacing[0], grid_spacing[1], grid_spacing[2])
# 设置数据格式 以及 number of components
image.AllocateScalars(vtk.VTK_INT, 1)

# 最后一步 将三维体中的每一个 scalar 数值填jin 上面创建的 image 对象中的对应位置
for x in range(dims[0]):
    for y in range(dims[1]):
        for z in range(dims[2]):
            image.SetScalarComponentFromDouble(x, y, z, 0, np_data[x, y, z])
"""

image = vtk.vtkNIFTIImageReader()
image.SetFileName("./data/image_lr.nii.gz")
image.Update()

# 对输入图像进行 Gaussian Smooth
filterImg = vtk.vtkImageGaussianSmooth()
# set input connection from the reader
filterImg.SetInputData(image.GetOutput())
filterImg.SetStandardDeviation(1, 1, 1)
filterImg.SetRadiusFactors(3, 3, 3)
filterImg.SetDimensionality(3)
filterImg.Update()

# 2. 利用 marching cubes 算法提取等值面  -------filter
iso = vtk.vtkMarchingCubes()
iso.SetInputData(filterImg.GetOutput())
iso.SetValue(0, 150)  # 设置第一个等值面的阈值  # 第一个参数为 i 个等值面， 第二个参数为该等值面的 value
iso.Update()

# 用laplacian smooth
Filter = vtk.vtkSmoothPolyDataFilter()
Filter.SetInputData(iso.GetOutput())
Filter.SetNumberOfIterations(20)
Filter.SetRelaxationFactor(1)
Filter.FeatureEdgeSmoothingOff()
Filter.BoundarySmoothingOff()
Filter.Update()


# 设置 mapper ------mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(Filter.GetOutput())

# 设置actor   ------actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1, 1, 0)  # 设置acotr 的颜色 和 光照 等属性
actor.GetProperty().SetAmbient(0.25)
actor.GetProperty().SetDiffuse(0.6)
actor.GetProperty().SetSpecular(1.0)

# 设置渲染器  -------renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)
ren.AddActor(actor)

# 设置窗口  -------renderWindow
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(512, 512)

# 设置交互器  -------interactor
iren = vtk.vtkRenderWindowInteractor()

# 启动
iren.SetRenderWindow(renWin)
iren.Initialize()
renWin.Render()
iren.Start()
