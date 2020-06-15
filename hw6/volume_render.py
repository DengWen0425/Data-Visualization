import vtk


file_name = "./data/backpack.vti"

# 读取数据文件  -----source/reader
img = vtk.vtkXMLImageDataReader()
img.SetFileName(file_name)

# 设置 ray cast 体渲染mapper
mapper = vtk.vtkGPUVolumeRayCastMapper()
mapper.SetInputConnection(img.GetOutputPort())
mapper.SetBlendModeToComposite()

# 设置颜色映射  scalar → RGB value
color_map = vtk.vtkColorTransferFunction()
color_map.AddRGBPoint(0,    0.5, 0.8, 1.0)
color_map.AddRGBPoint(15000, 0.0, 0.0, 0.55)

# 设置透明度映射 scalar → opacity
opacity_map = vtk.vtkPiecewiseFunction()
opacity_map .AddPoint(0, 0.0)
opacity_map .AddPoint(15000, 1)

# 将以上属性添加进 volume property 中
vol_property = vtk.vtkVolumeProperty()
vol_property.SetColor(color_map)
vol_property.SetScalarOpacity(opacity_map)

# 其他属性设置
vol_property.SetInterpolationTypeToLinear()  # 插值方法为线性插值
vol_property.ShadeOn()  # 阴影开启
vol_property.SetAmbient(0.)  # 环境光系数
vol_property.SetDiffuse(0.9)  # 散射光系数
vol_property.SetSpecular(0.5)  # 反射光系数

# 设置 3D 对象 --- 类似之前的actor
volume = vtk.vtkVolume()
volume.SetMapper(mapper)
volume.SetProperty(vol_property)

# 渲染器
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)
ren.AddVolume(volume)
light = vtk.vtkLight()
light.SetColor(1,1,1) #光的颜色
#设置灯光类型为相机灯光，灯光会自动随相机移动
light.SetLightTypeToCameraLight()
ren.AddLight(light)

# 渲染器窗口
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(512, 512)

# 设置交互器
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# 启动
iren.Initialize()
renWin.Render()
iren.Start()

