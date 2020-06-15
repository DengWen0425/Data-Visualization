import vtk

# the process of following code:
# source/reader → filter → mapper → actor → renderer → renderWindow → interactor

file_name = "./data/backpack.vti"

# 读取数据文件  -----source/reader
img = vtk.vtkXMLImageDataReader()
img.SetFileName(file_name)

# 利用 marching cubes 算法提取等值面  -------filter
iso = vtk.vtkMarchingCubes()
iso.SetInputConnection(img.GetOutputPort())
iso.SetValue(0, 1300)  # 设置第一个等值面的阈值  
# 第一个参数为 i 个等值面， 第二个参数为该等值面的 value

stripper = vtk.vtkStripper()  # 通过vtkStripper在等值面上产生纹理或三角面片
stripper.SetInputConnection(iso.GetOutputPort())

# 设置 mapper ------mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())

# 设置actor   ------actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0., 0.75, 1.)  #  设置acotr 的颜色 和 光照 等属性
actor.GetProperty().SetAmbient(0.25)
actor.GetProperty().SetDiffuse(0.6)
actor.GetProperty().SetSpecular(1)

# 设置渲染器  -------renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)
ren.AddActor(actor)

# 设置窗口  -------renderWindow
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(512, 512)

# 设置 Interactor  -------interactor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()
renWin.Render()
iren.Start()
