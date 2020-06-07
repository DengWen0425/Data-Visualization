import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Tools import *
from tran import Pic_local_affine
from tran import LocalAffine
from tran import base_LocalAffine


class Interface(QWidget):
  def __init__(self):
    super().__init__()
    self._diaheight = 800 # 窗口高度
    self._diawidth = 800 # 窗口宽度
    # 初始化参数
    self.isSource = False#是否添加源图片
    self.isTarget = False#是否添加狒狒图片
    self.clickOnST = ''
    self.source_txt = ''#源图片坐标输出
    self.target_txt = ''#狒狒图片左边输出
    self.sourceIndex = []#源图片坐标
    self.targetIndex = []#狒狒图片坐标
    self.initUI()#初始化界面

  @pyqtSlot(bool)
  # 打开狒狒的图片
  def get_source_pic_btn_clicked(self, checked):
    self.targetIndex = []
    self.targetfile = QFileDialog.getOpenFileName(self, "OpenFile", ".","Image Files(*.jpg *.jpeg *.png)")[0]
    print('targetfile', self.targetfile)

    if self.targetfile != '':
      # 读取图片
      self.image_t = QImage(self.targetfile) 

      # 展示图片
      self.imageView_target.setPixmap(QPixmap.fromImage(self.image_t)) 
      self.resize(self.image_t.width(), self.image_t.height())

      # 图片大小
      print('t_size:', self.image_t.width(), self.image_t.height()) 
      print('t_size_real:', self.imageView_target.width(), self.imageView_target.height())
      
      #读入源文件
      self.isTarget = True

  @pyqtSlot(bool)
  # 打开需要改变的图片
  def get_target_pic_btn_clicked(self, checked):
    self.sourceIndex = []
    self.sourcefile = QFileDialog.getOpenFileName(self, "OpenFile", ".","Image Files(*.jpg *.jpeg *.png)")[0]
    print('sourcefile', self.sourcefile)
    if self.sourcefile != '':
      # 读取图片
      self.image_s = QImage(self.sourcefile)
      
      # 显示图片
      self.imageView_source.setPixmap(QPixmap.fromImage(self.image_s))
      self.resize(self.image_s.width(), self.image_s.height())

      # 图片大小
      print('s_size:', self.image_s.width(), self.image_s.height())
      print('s_size_real:', self.imageView_source.width(), self.imageView_source.height())

      self.isSource = True

  @pyqtSlot(bool)
  # 按键——狒狒的图像
  def get_target_pic_btn(self):
    #图片显示
    self.imageView_target = QLabel("add a target image file")           # 得到一个QLabel的实例，负责显示消息以及图片
    self.imageView_target.setAlignment(Qt.AlignCenter)                  # 设置QLabel居中显示
    self.open_target = QPushButton("open target picture")                # 实例化一个名为"open"的按钮，并将它保存在类成员open中，负责获取图片的路径，并在QLabel中显示
    self.open_target.clicked.connect(self.get_source_pic_btn_clicked)   # 信号与槽的连接
    self.vlayout_target = QVBoxLayout()
    self.vlayout_target.addWidget(self.imageView_target)
    # 鼠标响应
    labelStatus = QLabel()
    labelStatus.setText(self.tr("affine Position:"))
    # 文字编辑框
    self.label_target = QTextEdit()
    self.label_target.setText(self.tr(""))
    self.vlayout_target.addWidget(labelStatus)
    self.vlayout_target.addWidget(self.label_target)
    self.vlayout_target.addWidget(self.open_target)

  @pyqtSlot(bool)
  # 按键——需要改变的图像
  def get_source_pic_btn(self):
    #图片显示
    self.imageView_source = QLabel("add a source image file")           # 得到一个QLabel的实例，负责显示消息以及图片
    self.imageView_source.setAlignment(Qt.AlignCenter)                  # 设置QLabel居中显示
    self.open_source = QPushButton("open source picture")               # 实例化一个名为"open"的按钮，并将它保存在类成员open中，负责获取图片的路径，并在QLabel中显示
    self.open_source.clicked.connect(self.get_target_pic_btn_clicked)   # 信号与槽的连接
    self.vlayout_source = QVBoxLayout()
    self.vlayout_source.addWidget(self.imageView_source)
    #鼠标响应
    labelStatus = QLabel()
    labelStatus.setText(self.tr("affine Position:"))
    # 文字编辑框
    self.label_source = QTextEdit()
    self.label_source.setText(self.tr(""))
    self.vlayout_source.addWidget(labelStatus)
    self.vlayout_source.addWidget(self.label_source)
    self.vlayout_source.addWidget(self.open_source)


  @pyqtSlot(bool)
  # 图像处理按键
  def get_tran_btn(self):
    #按钮
    labelSubmit = QLabel()
    self.submit_btn = QPushButton("TRANSFORM!")
    self.submit_btn.setToolTip('<b>Make sure there are 16 points</b>')#浮窗提示
    self.submit_btn.clicked.connect(self.on_btn_submit_btn_clicked)

    self.hlayout_submit = QHBoxLayout()
    self.hlayout_submit.addWidget(labelSubmit)
    self.hlayout_submit.addWidget(self.submit_btn)
  
  @pyqtSlot(bool)
  # 图像处理过程
  def on_btn_submit_btn_clicked(self, checked):
    sourcenp = [np.array(self.sourceIndex[:4]),np.array(self.sourceIndex[4:8]),np.array(self.sourceIndex[8:11]),np.array(self.sourceIndex[11:])]
    targetnp = [np.array(self.targetIndex[:4]),np.array(self.targetIndex[4:8]),np.array(self.targetIndex[8:11]),np.array(self.targetIndex[11:])]
    pic_aff = Pic_local_affine(self.sourcefile, targetnp, sourcenp, (256, 256))
    #pic_aff = Pic_local_affine('./data/human1.jpg', points_rigid, human_points, (256, 256))
    # 调用写好的局部仿射及反向映射实现人脸到狒狒脸的转换
    new_pic = pic_aff.derive_new_picture()
    plt.imshow(new_pic, cmap='gray')
    plt.savefig('.//result//generate.png')
    self.image = QImage('.//result//generate.png')
    self.imageView_target.setPixmap(QPixmap.fromImage(self.image))
    self.resize(self.image.width(), self.image.height())
    print(self.image.width(), self.image.height())

  @pyqtSlot(bool)
  # 操作提示
  def get_remainder(self):
    #文本信息
    labelSubmit = QLabel()
    labelSubmit.setText(self.tr("please clicke  4 pointes for each eyes  3 pointes for nose  5 pointes for mouths"))
    self.vlayout_remainder = QVBoxLayout()
    self.vlayout_remainder.addWidget(labelSubmit)
    self.vlayout_remainder.addWidget(labelSubmit)
    self.vlayout_remainder.addWidget(self.submit_btn)


  def initUI(self):
  # 从屏幕上（100，100）位置开始显示一个800 * 800的界面（宽800，高800）
    self.setGeometry(100, 100, 800, 800)
    self.setMinimumHeight(self._diaheight)
    self.setMinimumWidth(self._diawidth)
    
    # 设置窗口字体\名称\图标
    QToolTip.setFont(QFont('SansSerif', 10))
    self.setWindowTitle('transform man-face to ape-face')
    #self.setWindowIcon(QtGui.QIcon('./data/ape-1.png'))
        
    #初始化窗口元素
    self.get_source_pic_btn()
    self.get_target_pic_btn()
    self.get_tran_btn()
    self.get_remainder()
    
    #横布局 两个图片处理 转换按钮
    hlayout = QHBoxLayout()
    hlayout.addLayout(self.vlayout_source)
    hlayout.addLayout(self.vlayout_target)
    hlayout.addLayout(self.hlayout_submit)
    
    #纵布局 提示与上述横布局
    vlayout = QVBoxLayout()
    vlayout.addLayout(hlayout)
    vlayout.addLayout(self.vlayout_remainder)


    #输出窗口
    self.label_source.setText(self.tr(""))
    self.setLayout(vlayout)
    self.show()

  def mousePressEvent(self, elemen):
    x = elemen.x()
    y = elemen.y()
    # 鼠标左键选点
    
    if elemen.buttons() == Qt.LeftButton:
      print('click:  ', x, y)
      
      # 如果所选点在source图片内：
      if self.isSource and x >= self.imageView_source.x() and y >= self.imageView_source.y() and x < self.imageView_source.x() + self.imageView_source.width() and y < self.imageView_source.y() + self.imageView_source.height():
      # 窗口的点和对应图片的点转换
        x_trans = x - self.imageView_source.x() - (self.imageView_source.width() - self.image_s.width()) // 2
        y_trans = y - self.imageView_source.y()
      
      # 存入列表
        k = len(self.sourceIndex)
        self.sourceIndex.append([y_trans,x_trans])
      
      # 将坐标输出在对应文本框内
        text = str(k) + "\t  x: {0},  y: {1}".format(x_trans, y_trans) + '\n'
        self.source_txt += text
        self.label_source.setText(self.source_txt)
        print(text)
     
      # 如果所选点在狒狒图像内
      elif self.isTarget and x >= self.imageView_target.x() and y >= self.imageView_target.y() and x < self.imageView_target.x() + self.imageView_target.width() and y < self.imageView_target.y() + self.imageView_target.height():
      
        # 窗口的点和对应图片的点转换
        x_trans = x - self.imageView_target.x() - (self.imageView_target.width() - self.image_t.width()) // 2
        y_trans = y - self.imageView_target.y()
      
        # 存入列表
        k = len(self.targetIndex)
        self.targetIndex.append([y_trans,x_trans])
      
        # 将坐标输出在对应文本框内
        text = str(k) + "\t  x: {0},  y: {1}".format(x_trans, y_trans) + '\n'
        self.target_txt += text
        self.label_target.setText(self.target_txt)
        print(text)


if __name__ == '__main__':
  app = QtWidgets.QApplication(sys.argv)
  ex = Interface()
  sys.exit(app.exec_())