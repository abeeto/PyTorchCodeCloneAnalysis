import json
import sys
import os
import time
import qtawesome
from PyQt5.Qt import *
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import numpy as np
from SSD_NeXt import *
from Transforms import *
import torch
import torchvision.transforms as transforms
import _thread
from threading import Thread
import re
import shutil

root = 'GUI'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MainWindow(QMainWindow):
    """
    主界面窗口
    """

    def __init__(self):
        super().__init__()
        if not os.path.exists('GUI'):
            os.makedirs('GUI')
        if os.path.exists('GUI\\Label\\detection.png'):
            self.setWindowIcon(QIcon('GUI\\Label\\detection.png'))
        self.InitUI()

    def InitUI(self):
        self.setWindowTitle('Object Detection')  # 界面标题
        self.setFixedSize(1200, 750)  # 界面大小不可变

        self.mainWidget = QtWidgets.QWidget()
        self.mainLayout = QtWidgets.QGridLayout()
        self.mainWidget.setLayout(self.mainLayout)

        self.leftWidget = QtWidgets.QWidget()  # 创建左侧部件
        self.leftWidget.setObjectName('left_widget')
        self.leftLayout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.leftWidget.setLayout(self.leftLayout)  # 设置左侧部件布局为网格

        self.rightFrame = QtWidgets.QFrame()  # 创建右侧部件
        self.rightFrame.setObjectName('right_frame')

        self.mainLayout.addWidget(self.leftWidget, 0, 0, 12, 3)  # 左侧部件在第0行第0列，占12行3列
        self.mainLayout.addWidget(self.rightFrame, 0, 3, 12, 9)  # 右侧部件在第0行第3列，占12行9列
        self.setCentralWidget(self.mainWidget)  # 设置窗口主部件

        self.leftClose = QPushButton(qtawesome.icon('fa.close', color='white'), "")  # 关闭按钮
        self.leftMini = QPushButton(qtawesome.icon('mdi.window-minimize', color='white'), "")  # 最小化按钮

        self.appName = QPushButton('Detection')  # 应用名称
        self.appName.setObjectName('left_label')
        self.support = QPushButton('其他')
        self.support.setObjectName('left_label')
        self.button1 = QPushButton(qtawesome.icon('fa.image', color='white'), '\t 打开图片 ')
        self.button2 = QPushButton(qtawesome.icon('ei.video', color='white'), '\t 打开视频 ')
        self.button3 = QPushButton(qtawesome.icon('fa.video-camera', color='white'), '\t打开摄像头')
        self.button4 = QPushButton(qtawesome.icon('mdi.monitor-screenshot', color='white'), '\t 检测画面 ')
        self.button5 = QPushButton(qtawesome.icon('msc.settings-gear', color='white'), "\t设置")
        self.button6 = QPushButton(qtawesome.icon('mdi.help-circle-outline', color='white'), "\t帮助")
        self.button1.setObjectName('left_button')
        self.button2.setObjectName('left_button')
        self.button3.setObjectName('left_button')
        self.button4.setObjectName('left_button')
        self.button5.setObjectName('left_button')
        self.button6.setObjectName('left_button')
        self.FormatWidget()

        self.leftLayout.addWidget(self.leftMini, 0, 0, 1, 1)
        self.leftLayout.addWidget(self.leftClose, 0, 2, 1, 1)
        self.leftLayout.addWidget(self.appName, 0, 1, 1, 1)
        self.leftLayout.addWidget(self.button1, 3, 0, 1, 3)
        self.leftLayout.addWidget(self.button2, 4, 0, 1, 3)
        self.leftLayout.addWidget(self.button3, 5, 0, 1, 3)
        self.leftLayout.addWidget(self.button4, 7, 0, 1, 3)
        self.leftLayout.addWidget(self.support, 9, 0, 1, 3)
        self.leftLayout.addWidget(self.button5, 10, 0, 1, 3)
        self.leftLayout.addWidget(self.button6, 11, 0, 1, 3)

        self.media_detector = MediaDetector()
        self.setting = Setting()
        self.media_detector.SetConfig(self.setting.ReadConfig())
        self.setting.SetValue(self.media_detector)
        self.help = Help()

        self.stack = QStackedLayout(self.rightFrame)  # 堆栈控件，放置对应功能的界面
        self.stack.addWidget(self.media_detector)  # 放入对应界面
        self.stack.addWidget(self.setting)
        self.stack.addWidget(self.help)

        # 槽函数绑定
        self.button1.clicked.connect(self.OnClickButton1)
        self.button2.clicked.connect(self.OnClickButton2)
        self.button3.clicked.connect(self.OnClickButton3)
        self.button4.clicked.connect(self.OnClickButton4)
        self.button5.clicked.connect(self.OnClickButton5)
        self.button6.clicked.connect(self.OnClickButton6)
        self.setting.affirm.clicked.connect(self.OnClickSettingAffirm)
        self.setting.cancel.clicked.connect(self.OnClickSettingCancel)

        self.leftClose.clicked.connect(self.close)
        self.leftMini.clicked.connect(self.showMinimized)
        self.leftClose.setFixedSize(40, 40)  # 设置关闭按钮的大小
        self.leftMini.setFixedSize(40, 40)  # 设置最小化按钮大小

        self.leftClose.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:10px;}QPushButton:hover{background:red;}''')
        self.leftMini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:10px;}QPushButton:hover{background:green;}''')

        self.leftWidget.setStyleSheet('''
                QPushButton{
                    border:none;
                    color:white;
                }
                QPushButton#left_label{
                    border:none;
                    border-bottom:1px solid white;
                    font-size:30px;
                    font-weight:700;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QLabel{
                    color:white;
                    font-size:18px;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QPushButton#left_button{
                    font-size:22px;
                }
                QPushButton#left_button_special{
                    font-size:22px;
                }
                QPushButton#left_button_special:hover{
                    background-color:black;
                    border-left:4px solid red;
                    border-right:4px solid red;
                    border-top:4px solid red;
                    border-bottom:4px solid red;
                    border-radius:10px;
                    font-weight:1000;
                    color:white;
                }
                QPushButton#left_button:hover{
                    background-color:black;
                    border-left:4px solid red;
                    border-right:4px solid red;
                    border-top:4px solid red;
                    border-bottom:4px solid red;
                    border-radius:10px;
                    font-weight:1000;
                    color:white;
                }
                QWidget#left_widget{
                    background:gray;
                    border-top:1px solid white;
                    border-bottom:1px solid white;
                    border-left:1px solid white;
                    border-top-left-radius:10px;
                    border-bottom-left-radius:10px;
                }
        ''')

        self.rightFrame.setStyleSheet('''
            QFrame#right_frame{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#right_lable{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "微软雅黑", Helvetica, Arial, sans-serif;
            }
        ''')

        self.setWindowOpacity(0.95)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.mainLayout.setSpacing(0)

        if os.path.exists(root + r'/Label/default.png'):
            default = cv2.imdecode(np.fromfile(root + r'/Label/default.png', dtype=np.uint8), -1)
            frame = Resize(default, size=(726, 915))
            self.media_detector.ShowFrame(frame)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mFlag = True
            self.mPosition = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.mFlag:
            self.move(QMouseEvent.globalPos() - self.mPosition)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.mFlag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def FormatWidget(self):
        """
        设置控件格式
        """
        sizePolicy = QSizePolicy()  # 设置按钮控件格式为水平垂直自动扩展
        sizePolicy.setVerticalPolicy(QSizePolicy.Expanding)
        sizePolicy.setHorizontalPolicy(QSizePolicy.Expanding)
        self.button1.setSizePolicy(sizePolicy)
        self.button2.setSizePolicy(sizePolicy)
        self.button3.setSizePolicy(sizePolicy)
        self.button4.setSizePolicy(sizePolicy)
        self.button5.setSizePolicy(sizePolicy)
        self.button6.setSizePolicy(sizePolicy)

    def OpenMediaFile(self, mode):
        if self.media_detector.detecting:
            return
        if mode == 1:
            # 图片
            img_file, img_type = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;*.png")
            if img_file != '':
                self.media_detector.detect_mode = 1
                self.media_detector.file_name = img_file
                frame = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
                self.media_detector.ShowFrame(frame)
                self.media_detector.save_button.setEnabled(False)

        elif mode == 2:
            # 视频
            img_file, img_type = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;*.avi")
            if img_file != '':
                self.media_detector.detect_mode = 2
                self.media_detector.file_name = img_file
                capture = cv2.VideoCapture(img_file)
                ret, frame = capture.read()
                self.media_detector.ShowFrame(frame)
                self.media_detector.save_button.setEnabled(False)

        elif mode == 3:
            # 摄像头
            try:
                self.media_detector.detect_mode = 3
                self.media_detector.file_name = 0
                self.media_detector.OnClickDetect()
            except:
                pass

    def OnClickButton1(self):
        """
        按钮1槽函数
        :return:
        """
        if self.stack.currentIndex() != 0:
            self.stack.setCurrentIndex(0)
        self.OpenMediaFile(mode=1)

    def OnClickButton2(self):
        """
        按钮2槽函数
        :return:
        """
        if self.stack.currentIndex() != 0:
            self.stack.setCurrentIndex(0)
        self.OpenMediaFile(mode=2)

    def OnClickButton3(self):
        """
        按钮3槽函数
        :return:
        """
        if self.stack.currentIndex() != 0:
            self.stack.setCurrentIndex(0)
        self.OpenMediaFile(mode=3)

    def OnClickButton4(self):
        """
        按钮4槽函数
        :return:
        """
        if self.stack.currentIndex() != 0:
            self.stack.setCurrentIndex(0)

    def OnClickButton5(self):
        """
        按钮5槽函数
        :return:
        """
        if self.stack.currentIndex() != 1:
            self.stack.setCurrentIndex(1)

    def OnClickButton6(self):
        """
        按钮6槽函数
        :return:
        """
        if self.stack.currentIndex() != 2:
            self.stack.setCurrentIndex(2)

    def OnClickSettingAffirm(self):
        self.setting.GetValue(self.media_detector)

    def OnClickSettingCancel(self):
        self.setting.SetValue(self.media_detector)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        try:
            if os.path.exists(self.media_detector.temp_root):
                shutil.rmtree(self.media_detector.temp_root)
        except:
            pass


class MediaDetector(QWidget):
    """播放器界面"""

    def __init__(self):
        super(MediaDetector, self).__init__()
        self.resize(1000, 1000)
        self.fps = 24
        self.detect_mode = 0  # 0为默认，1为图像，2为视频，3为实时
        self.file_name = ''
        self.save_root = r'GUI\Record'
        self.temp_root = r'GUI\Temp'
        self.auto_save = True
        self.detecting = False
        self.pause = False
        self.stop = False
        self.IOU_threshold = 0.45
        self.conf_threshold = 0.4
        self.cv2_label = [cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2]  # 标签的字体、大小、颜色、粗细
        self.show_classes = True
        self.line_width = 2
        self.show_score = True
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # 颜色
        self.model_type = 1
        self.InitUI()

    def InitUI(self):
        self.detect_button = QPushButton(qtawesome.icon('fa.object-group', color='black'), '检测')
        self.pause_button = QPushButton(qtawesome.icon('fa5.pause-circle', color='black'), '暂停')
        self.stop_button = QPushButton(qtawesome.icon('fa5.stop-circle', color='black'), '终止')  # 终止键
        self.save_button = QPushButton(qtawesome.icon('fa.save', color='black'), '保存')  # 保存键
        self.canvas = QLabel()  # 画布，放置画面
        self.canvas.setScaledContents(True)
        self.waiting = WaitingWidget()

        self.Format()  # 设置文本格式

        self.detect_button.clicked.connect(self.OnClickDetect)
        self.pause_button.clicked.connect(self.OnClickPause)
        self.stop_button.clicked.connect(self.OnClickStop)
        self.save_button.clicked.connect(self.OnClickSave)

        tools = QHBoxLayout()  # 水平布局的工具栏
        tools.addWidget(self.detect_button)
        tools.addWidget(self.pause_button)
        tools.addWidget(self.stop_button)
        tools.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        tools.addWidget(self.save_button)

        layout = QVBoxLayout()  # 垂直布局，上侧放置工具栏，下侧为画面
        layout.addLayout(tools)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.setStyleSheet('''
                QPushButton{
                    border:none;
                    color:#707070;
                    font-size:20px;
                    font-weight:500;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QPushButton:hover{
                    color:red;
                    font-size:21px;
                    font-weight:1000;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
        ''')
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def Format(self):
        """
        控件格式设置
        """
        sizePolicy = QSizePolicy()  # 设置控件格式为水平自适应或扩展
        sizePolicy.setHorizontalPolicy(QSizePolicy.Expanding)
        sizePolicy.setVerticalPolicy(QSizePolicy.Fixed)
        self.detect_button.setSizePolicy(sizePolicy)
        self.pause_button.setSizePolicy(sizePolicy)
        self.stop_button.setSizePolicy(sizePolicy)
        self.save_button.setSizePolicy(sizePolicy)

    def ShowFrame(self, frame):
        """
        展示图像
        """
        frame = Resize(frame, size=(self.height(), self.width()))
        rbg_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        qt_img = QtGui.QImage(rbg_img.data, rbg_img.shape[1], rbg_img.shape[0], QtGui.QImage.Format_RGBA8888)
        self.canvas.setPixmap(QtGui.QPixmap.fromImage(qt_img))
        self.canvas.show()
        cv2.waitKey(1)

    def DetectThread(self):
        """
        检测线程，检测任务主流程
        """
        if not os.path.exists(self.file_name):
            QMessageBox.warning(QWidget(), '警告', '文件路径错误，请重新选择文件', QMessageBox.Yes)
            return
        self.detect_button.setEnabled(False)
        self.detect_button.setText('检测中...')
        self.detecting = True
        self.save_button.setEnabled(False)
        if not os.path.exists(self.temp_root):
            os.makedirs(self.temp_root)
        self.temp_file = os.path.join(self.temp_root, os.path.split(self.file_name)[1])
        if self.detect_mode == 1:
            # 图片
            frame = cv2.imdecode(np.fromfile(self.file_name, dtype=np.uint8), -1)
            self.DetectCore(frame)
            cv2.imwrite(self.temp_file, frame)

        elif self.detect_mode == 2 or self.detect_mode == 3:
            # 视频或摄像头
            capture = cv2.VideoCapture(self.file_name)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(self.temp_file, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            while True:
                if self.stop:
                    self.Restore()
                    break
                ret, frame = capture.read()
                if ret:
                    if self.pause:
                        self.ShowFrame(frame)
                    else:
                        self.DetectCore(frame)
                    out.write(frame)
            out.release()

        self.AutoSave()
        self.detecting = False
        self.save_button.setEnabled(True)
        self.detect_button.setEnabled(True)
        self.detect_button.setText('检测')

    def DetectCore(self, frame):
        """
        核心检测部分，包含检测目标和绘制结果
        """
        # 输入图片，进行预测
        height, width = frame.shape[0], frame.shape[1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = self.transforms_image(rgb)
        input_frame = self.transforms_all(input_frame, None, True)
        with torch.no_grad():
            objects = self.model.Predict(input_frame.unsqueeze(0).to(device),
                                         IOU_threshold=self.IOU_threshold, conf_threshold=self.conf_threshold)[0]  # 预测

        if objects.shape[0] > 0:
            objects = self.transforms_all.Recover(size=(height, width), anchors=objects).cpu()

            # 格式调整
            bboxes = (objects[:, :4] * torch.tensor([width, height, width, height]))  # 转化为像素坐标
            left_top = bboxes[:, :2].round().int().tolist()
            right_bottom = bboxes[:, 2:4].round().int().tolist()
            if self.show_classes:
                labels = [self.classes[i] for i in objects[:, 4].int().tolist()]  # 转化为字符串形式的类别标签
                cls_colors = [self.colors[i % len(self.colors)] for i in objects[:, 4].int().tolist()]
            else:
                cls_colors = [random.choice(self.colors) for _ in range(objects.shape[0])]
            scores = (objects[:, 5] * 100).round().int().tolist()

            # 绘制目标
            for obj in range(objects.shape[0]):
                cv2.rectangle(frame, left_top[obj], right_bottom[obj], cls_colors[obj], self.line_width)

                text = ''
                if self.show_classes:
                    text += f'{labels[obj]} '
                if self.show_score:
                    text += f'{scores[obj]}%'
                if text != '':
                    left_top[obj][1] -= self.line_width * 3
                    cv2.putText(frame, text, left_top[obj],
                                self.cv2_label[0], self.cv2_label[1], self.cv2_label[2], self.cv2_label[3])
        self.ShowFrame(frame)

    def Restore(self):
        """
        按钮状态复原
        """
        self.stop_button.setEnabled(False)
        self.stop = False
        self.pause_button.setEnabled(False)
        self.pause = False
        self.pause_button.setText('暂停')
        self.pause_button.setIcon(qtawesome.icon('fa5.pause-circle', color='black'))

    def InitModel(self):
        """
        初始化模型和数据预处理
        """
        if self.model_type == 1:
            self.model = SSD_NeXt(num_classes=20, cfg=SSD_NeXt_cfg).to(device)
            self.model.load_state_dict(torch.load(r'weights/model_SSD-NeXt_VOC.pth'))
            self.classes = ['person',
                            'car', 'bus', 'bicycle', 'motorbike', 'aeroplane', 'boat', 'train',
                            'chair', 'sofa', 'diningtable', 'tvmonitor', 'bottle', 'pottedplant',
                            'cat', 'dog', 'cow', 'horse', 'sheep', 'bird']
        elif self.model_type == 2:
            self.model = SSD_NeXt(num_classes=2, cfg=SSD_NeXt_cfg).to(device)
            self.model.load_state_dict(torch.load(r'weights/model_SSD-NeXt_KITTI.pth'))
            self.classes = ['person', 'vehicle']
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=mean, std=std)])
        self.transforms_all = LetterBoxResize(size=(self.model.h, self.model.w))

    def OnClickDetect(self):
        """
        检测键槽函数
        """
        if self.detecting:
            return
        try:
            # 创建新线程进行检测
            _thread.start_new_thread(self.DetectThread, ())
        except:
            print("线程创建失败")

    def OnClickPause(self):
        """
        暂停键槽函数
        """
        self.pause = not self.pause
        if self.pause:
            self.pause_button.setText('继续')
            self.pause_button.setIcon(qtawesome.icon('msc.debug-continue', color='black'))
        else:
            self.pause_button.setText('暂停')
            self.pause_button.setIcon(qtawesome.icon('fa5.pause-circle', color='black'))

    def OnClickStop(self):
        """
        终止键槽函数
        """
        self.stop = True

    def OnClickSave(self):
        """
        保存键槽函数
        """
        if os.path.isfile(self.temp_file):
            # 确保检测后的临时文件存在
            target_file, _ = QFileDialog.getSaveFileName(self, '选择保存路径', self.file_name, "")
            if target_file != '':
                # 使用多线程异步另存为新文件
                Thread(target=shutil.copy, args=[self.temp_file, target_file]).start()
        else:
            QMessageBox.warning(QWidget(), '警告', '保存失败，原始文件已丢失', QMessageBox.Yes)

    def AutoSave(self):
        """
        自动保存到指定文件夹
        """
        if self.auto_save:
            if not os.path.exists(self.save_root):
                os.makedirs(self.save_root)
            save_file = self.AdjustFileName(os.path.join(self.save_root, os.path.split(self.file_name)[1]))
            # 将临时文件复制到保存文件内
            Thread(target=shutil.copy, args=[self.temp_file, save_file]).start()

    def AdjustFileName(self, path):
        """
        调整文件名，有重名则在后面加数字
        """
        directory, file_name = os.path.split(path)
        while os.path.isfile(path):
            pattern = '(\d+)\)\.'
            if re.search(pattern, file_name) is None:
                file_name = file_name.replace('.', '(1).')
            else:
                current_number = int(re.findall(pattern, file_name)[-1])
                new_number = current_number + 1
                file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
            path = os.path.join(directory, file_name)
        return path

    def SetConfig(self, config):
        """
        设置参数
        """
        if isinstance(config, dict):
            if 'model_type' in config:
                self.model_type = config['model_type']
            if 'IOU_threshold' in config:
                self.IOU_threshold = config['IOU_threshold']
            if 'conf_threshold' in config:
                self.conf_threshold = config['conf_threshold']
            if 'cv2_label' in config:
                self.cv2_label = config['cv2_label']
                self.cv2_label[2] = tuple(self.cv2_label[2])
            if 'show_classes' in config:
                self.show_classes = config['show_classes']
            if 'show_score' in config:
                self.show_score = config['show_score']
            if 'line_width' in config:
                self.line_width = config['line_width']
            if 'save_root' in config:
                self.save_root = config['save_root']
            if 'auto_save' in config:
                self.auto_save = config['auto_save']
        self.InitModel()


class Setting(QWidget):
    """设置界面"""

    def __init__(self):
        super(Setting, self).__init__()
        self.resize(750, 750)
        self.fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
                      cv2.FONT_HERSHEY_TRIPLEX]
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0),
                       (255, 255, 255)]
        self.InitUI()

    def InitUI(self):
        sizePolicy = QSizePolicy()  # 设置控件格式为固定
        sizePolicy.setHorizontalPolicy(QSizePolicy.Fixed)
        sizePolicy.setVerticalPolicy(QSizePolicy.Fixed)

        # 选择模型
        self.b_model1 = QRadioButton('SSD-NeXt-VOC')
        self.b_model2 = QRadioButton('SSD-NeXt-KITTI')
        self.bg_models = QButtonGroup()
        self.bg_models.addButton(self.b_model1, 11)
        self.bg_models.addButton(self.b_model2, 12)
        models_box = QHBoxLayout()
        models_box.setContentsMargins(50, 10, 50, 0)
        models_box.addWidget(self.b_model1)
        models_box.addWidget(self.b_model2)
        models_box_title = QHBoxLayout()
        models_box_title.setContentsMargins(0, 20, 0, 0)
        label = QLabel('模型设置')
        label.setSizePolicy(sizePolicy)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        models_box_title.addWidget(label)
        models_box_title.addWidget(line)

        # 阈值
        self.s_iou = QSlider()
        self.s_conf = QSlider()
        self.s_iou_value = QLabel()
        self.s_conf_value = QLabel()
        threshold_box = QHBoxLayout()
        threshold_box.setContentsMargins(50, 10, 50, 0)
        threshold_box.addWidget(QLabel('IoU阈值'))
        threshold_box.addWidget(self.s_iou)
        threshold_box.addWidget(self.s_iou_value)
        threshold_box.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        threshold_box.addWidget(QLabel('置信度阈值'))
        threshold_box.addWidget(self.s_conf)
        threshold_box.addWidget(self.s_conf_value)
        threshold_box_title = QHBoxLayout()
        threshold_box_title.setContentsMargins(0, 20, 0, 0)
        label = QLabel('阈值设置')
        label.setSizePolicy(sizePolicy)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        threshold_box_title.addWidget(label)
        threshold_box_title.addWidget(line)

        # 标签字体
        self.c_font = QComboBox()
        self.s_font_size = QSlider()
        self.s_font_size.setOrientation(Qt.Horizontal)
        self.c_font_color = QComboBox()
        self.s_font_width = QSlider()
        self.s_font_width.setOrientation(Qt.Horizontal)
        self.b_show_cls = QCheckBox('展示类别')
        self.b_show_score = QCheckBox('展示置信度')
        font_box1 = QHBoxLayout()
        font_box1.setContentsMargins(50, 10, 50, 0)
        font_box1.addWidget(QLabel('字体类型'))
        font_box1.addWidget(self.c_font)
        font_box1.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        font_box1.addWidget(QLabel('字体大小'))
        font_box1.addWidget(self.s_font_size)

        font_box2 = QHBoxLayout()
        font_box2.setContentsMargins(50, 10, 50, 0)
        font_box2.addWidget(QLabel('字体颜色'))
        font_box2.addWidget(self.c_font_color)
        font_box2.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        font_box2.addWidget(QLabel('字体粗细'))
        font_box2.addWidget(self.s_font_width)

        font_box3 = QHBoxLayout()
        font_box3.setContentsMargins(50, 10, 50, 0)
        font_box3.addWidget(self.b_show_cls)
        font_box3.addWidget(self.b_show_score)

        font_box_title = QHBoxLayout()
        font_box_title.setContentsMargins(0, 20, 0, 0)
        label = QLabel('标签设置')
        label.setSizePolicy(sizePolicy)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        font_box_title.addWidget(label)
        font_box_title.addWidget(line)

        # 边界框
        self.s_box_width = QSlider()
        self.s_box_width.setOrientation(Qt.Horizontal)
        bbox_box = QHBoxLayout()
        bbox_box.setContentsMargins(50, 10, 50, 0)
        bbox_box.addWidget(QLabel('线条粗细'))
        bbox_box.addWidget(self.s_box_width)
        bbox_box.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        bbox_box_title = QHBoxLayout()
        bbox_box_title.setContentsMargins(0, 20, 0, 0)
        label = QLabel('边界框设置')
        label.setSizePolicy(sizePolicy)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        bbox_box_title.addWidget(label)
        bbox_box_title.addWidget(line)

        # 历史记录根目录
        self.l_dir = QLineEdit()
        self.l_dir.setReadOnly(True)
        self.b_dir = QPushButton('选择路径')
        self.b_auto_save = QCheckBox('自动保存')
        record_box = QHBoxLayout()
        record_box.setContentsMargins(50, 10, 50, 0)
        record_box.addWidget(QLabel('保存目录'))
        record_box.addWidget(self.l_dir)
        record_box.addWidget(self.b_dir)
        record_box.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        record_box2 = QHBoxLayout()
        record_box2.setContentsMargins(50, 10, 50, 0)
        record_box2.addWidget(self.b_auto_save)
        record_box2.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        record_box_title = QHBoxLayout()
        record_box_title.setContentsMargins(0, 20, 0, 0)
        label = QLabel('检测记录设置')
        label.setSizePolicy(sizePolicy)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        record_box_title.addWidget(label)
        record_box_title.addWidget(line)

        # 确认和取消
        self.affirm = QPushButton(qtawesome.icon('mdi.check-circle', color='black'), '确认')
        self.cancel = QPushButton(qtawesome.icon('mdi.close-circle', color='black'), '取消')
        tail_box = QHBoxLayout()
        tail_box.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        tail_box.addWidget(self.affirm)
        tail_box.addWidget(self.cancel)

        self.Format()
        self.s_iou.valueChanged.connect(self.ShowIOU)
        self.s_conf.valueChanged.connect(self.ShowConf)
        self.b_dir.clicked.connect(self.ChooseDir)

        # 总体布局
        layout = QVBoxLayout()  # 垂直布局

        layout.addLayout(models_box_title)
        layout.addLayout(models_box)

        layout.addLayout(threshold_box_title)
        layout.addLayout(threshold_box)

        layout.addLayout(font_box_title)
        layout.addLayout(font_box1)
        layout.addLayout(font_box2)
        layout.addLayout(font_box3)

        layout.addLayout(bbox_box_title)
        layout.addLayout(bbox_box)

        layout.addLayout(record_box_title)
        layout.addLayout(record_box)
        layout.addLayout(record_box2)

        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addLayout(tail_box)

        self.setLayout(layout)

        self.setStyleSheet('''
                QPushButton{
                    border:none;
                    color:#707070;
                    font-size:20px;
                    font-weight:500;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QPushButton:hover{
                    color:red;
                    font-size:21px;
                    font-weight:1000;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QLabel{
                    color:black;
                    font-size:18px;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QRadioButton{
                    color:black;
                    font-size:18px;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QCheckBox{
                    color:black;
                    font-size:18px;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QComboBox{
                    color:black;
                    font-size:18px;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
                QLineEdit{
                    color:black;
                    font-size:18px;
                    font-family: "微软雅黑", Helvetica, Arial, sans-serif;
                }
        ''')

    def Format(self):
        """
        设置控件格式
        """
        # 阈值控件
        self.s_iou.setOrientation(Qt.Horizontal)
        self.s_iou.setMinimum(10)
        self.s_iou.setMaximum(95)
        self.s_iou.setPageStep(5)
        self.s_iou.setSingleStep(5)
        self.s_iou.setTickPosition(QSlider.TicksBelow)
        self.s_conf.setOrientation(Qt.Horizontal)
        self.s_conf.setMinimum(10)
        self.s_conf.setMaximum(95)
        self.s_conf.setPageStep(5)
        self.s_conf.setSingleStep(5)
        self.s_conf.setTickPosition(QSlider.TicksBelow)

        # 标签控件
        self.c_font.addItems(['简约', '普通', '复杂', '精美'])
        self.s_font_size.setMinimum(5)
        self.s_font_size.setMaximum(50)
        self.s_font_size.setPageStep(5)
        self.s_font_size.setSingleStep(5)
        self.s_font_size.setTickPosition(QSlider.TicksBelow)
        self.c_font_color.addItems(['红色', '绿色', '蓝色', '黄色', '紫色', '青色', '黑色', '白色'])
        self.s_font_width.setMinimum(1)
        self.s_font_width.setMaximum(5)
        self.s_font_width.setPageStep(1)
        self.s_font_width.setSingleStep(1)
        self.s_font_width.setTickPosition(QSlider.TicksBelow)

        # 边界框控件
        self.s_box_width.setMinimum(1)
        self.s_box_width.setMaximum(5)
        self.s_box_width.setPageStep(1)
        self.s_box_width.setSingleStep(1)
        self.s_box_width.setTickPosition(QSlider.TicksBelow)

    def SetValue(self, media_detector):
        """
        设置控件显示参数
        """
        # 模型设置
        if media_detector.model_type == 1:
            self.b_model1.setChecked(True)
        else:
            self.b_model2.setChecked(True)

        # 阈值设置
        self.s_iou.setValue(int(media_detector.IOU_threshold * 100))
        self.s_conf.setValue(int(media_detector.conf_threshold * 100))

        # 标签设置 [cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2]  # 标签的字体、大小、颜色、粗细
        font = media_detector.cv2_label
        self.c_font.setCurrentIndex(self.fonts.index(font[0]))
        self.s_font_size.setValue(int(font[1] * 10))
        self.c_font_color.setCurrentIndex(self.colors.index(font[2]))
        self.s_font_width.setValue(font[3])
        self.b_show_cls.setChecked(media_detector.show_classes)
        self.b_show_score.setChecked(media_detector.show_score)

        # 边界框设置
        self.s_box_width.setValue(media_detector.line_width)

        # 历史记录设置
        self.l_dir.setText(media_detector.save_root)
        self.b_auto_save.setChecked(media_detector.auto_save)

        self.WriteConfig(media_detector)

    def GetValue(self, media_detector):
        """
        获取设定参数并调整系统参数
        :param media_detector:
        :return:
        """
        # 模型设置
        if self.b_model1.isChecked() and media_detector.model_type != 1:
            media_detector.model_type = 1
            media_detector.InitModel()
        elif self.b_model2.isChecked() and media_detector.model_type != 2:
            media_detector.model_type = 2
            media_detector.InitModel()

        # 阈值设置
        media_detector.IOU_threshold = self.s_iou.value() / 100
        media_detector.conf_threshold = self.s_conf.value() / 100

        # 标签设置 [cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2]  # 标签的字体、大小、颜色、粗细
        media_detector.cv2_label[0] = self.fonts[self.c_font.currentIndex()]
        media_detector.cv2_label[1] = self.s_font_size.value() / 10
        media_detector.cv2_label[2] = self.colors[self.c_font_color.currentIndex()]
        media_detector.cv2_label[3] = self.s_font_width.value()
        media_detector.show_classes = self.b_show_cls.isChecked()
        media_detector.show_score = self.b_show_score.isChecked()

        # 边界框设置
        media_detector.line_width = self.s_box_width.value()

        # 历史记录设置
        media_detector.save_root = self.l_dir.text()
        media_detector.auto_save = self.b_auto_save.isChecked()

        self.WriteConfig(media_detector)

    def ShowIOU(self):
        self.s_iou_value.setText(str(self.s_iou.value()))

    def ShowConf(self):
        self.s_conf_value.setText(str(self.s_conf.value()))

    def ChooseDir(self):
        root = QFileDialog.getExistingDirectory(self, "选择路径", ".")
        if root != '' and os.path.isdir(root):
            self.l_dir.setText(root)

    def WriteConfig(self, media_detector):
        """
        写入配置参数
        """
        config = {
            'model_type': media_detector.model_type,
            'IOU_threshold': media_detector.IOU_threshold,
            'conf_threshold': media_detector.conf_threshold,
            'cv2_label': media_detector.cv2_label,
            'show_classes': media_detector.show_classes,
            'show_score': media_detector.show_score,
            'line_width': media_detector.line_width,
            'save_root': media_detector.save_root,
            'auto_save': media_detector.auto_save
        }
        json_str = json.dumps(config, indent=4)
        if not os.path.exists('GUI'):
            os.makedirs('GUI')
        with open(os.path.join('GUI', 'config.json'), 'w') as json_file:
            json_file.write(json_str)

    def ReadConfig(self):
        """
        读取配置参数
        """
        config_path = os.path.join('GUI', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as json_file:
                json_str = json_file.read()
            config = json.loads(json_str)
            return config


class Help(QWidget):
    """
    帮助界面
    """

    def __init__(self):
        super(Help, self).__init__()
        self.resize(750, 750)
        self.InitUI()

    def InitUI(self):
        self.text = QLabel()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 30)
        layout.addWidget(self.text)
        self.setLayout(layout)
        self.SetText()
        self.setStyleSheet('''
            QLabel{
                color:black;
                font-size:20px;
                font-weight:500;
                font-family:'微软雅黑';  
            }
        ''')
        self.text.setAlignment(Qt.AlignTop)

    def SetText(self):
        html = """
            <html><head/>
                <body>
    
                    <p align=\"center\">使用方法</p>
                    <p>在左侧导航栏选择打开图像文件，然后在右侧界面点击“检测”即可</p>
                    <p>对视频帧检测时点击“暂停”可临时停止检测，但不影响视频播放，点击终止可完成检测</p>
                    <p>检测完成的图片或视频均可另存至自定义位置</p>
                    <p>设置界面可进行参数设置，点击“检测画面”即可回到目标检测界面</p>
                    <p>受设备IO影响，打开软件和加载新模型后首次检测可能会有点慢</p>
                    
                    <p align=\"center\"><hr/></p>        
                    <p align=\"center\">参数设置</p>
                    <p>SSD-NeXt-VOC能够识别20类常见目标</p>
                    <p>SSD-NeXt-KITTI仅能识别机动车和人，但精度更高</p>
                    <p>提示：更换模型时会重新加载模型，可能耗时稍长</p>
                    <p>IOU阈值指NMS算法去除冗余框的IOU阈值，阈值越低，重叠框去除越严格</p>
                    <p>置信度阈值指保留目标的置信度下限，低于阈值的检测结果将不被展示</p>
                    <p>其余参数可以调整边界框的展示形式、检测结果的保存状态等</p>

                </body>
            </html>
        """
        self.text.setText(html)


class WaitingWidget(QWidget):
    """等待界面"""

    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 100)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.label = QLabel('正在加载模型，请稍后...')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.raise_()

        v_box = QVBoxLayout()
        v_box.addWidget(self.label)

        self.setStyleSheet('''
            QWidget{
                background-color: #D3D3D3;
                border-color:black;
                border-style:solid;
                border-width:2px;
            }
            QLabel{
                color: black;
                font-size: 22px;
                font-family: "微软雅黑", Helvetica, Arial, sans-serif;
            }
        ''')
        self.setLayout(v_box)

    def open(self, text):
        self.label.setText(text)
        self.show()


def Resize(image, size):
    """Letterbox Resize"""
    # 计算宽和高的缩放比例，选择最小的那个进行等比例缩放，并在另一维度填充至指定size
    h, w = image.shape[0], image.shape[1]
    ratio_h = size[0] / h
    ratio_w = size[1] / w

    if ratio_h < ratio_w:
        ratio = ratio_h
        dim = 1  # 待填充的维度
    else:
        ratio = ratio_w
        dim = 0
    image = cv2.resize(image, (round(w * ratio), round(h * ratio)))  # 等比例缩放
    padding_size = size[dim] - image.shape[dim]

    padding_1 = round(padding_size / 2)
    padding_2 = padding_size - padding_1

    # 白色填充
    if dim == 1:
        image = cv2.copyMakeBorder(image, 0, 0, padding_1, padding_2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        image = cv2.copyMakeBorder(image, padding_1, padding_2, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
