import sys
import gc
from Forward_Window import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import numpy as np
import datetime
import collections
from learning.mnist.model.all_methods import neural_test_forward
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QDialog, QMessageBox, QPushButton, QTableView, QLabel, QHeaderView, QVBoxLayout
class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    signal = pyqtSignal(dict, str)
    all_signal = pyqtSignal(list, list)
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
# 定义槽函数
    def forward_calculate(self):
        starttime = datetime.datetime.now()
        lines = self.lineEdit.text().split('/')
        filepath = ''
        for i in range(len(lines)):
            if i == 0:
                filepath = lines[i]
            else:
                filepath = filepath + '\\' + lines[i]
        print(filepath)
        neural_test_forward(filepath)

        endtime = datetime.datetime.now()
        seconds = (endtime - starttime).seconds
        start = starttime.strftime('%Y-%m-%d %H:%M')
            # 100 秒
            # 分钟
        minutes = seconds // 60
        second = seconds % 60
        print((endtime - starttime))
        timeStr = str(minutes) + '分钟' + str(second) + "秒"
        print("程序从 " + start + ' 开始运行,运行时间为：' + timeStr)
        reply = QMessageBox.information(self, "提示", "正向计算完成,用时"+timeStr, QMessageBox.Yes | QMessageBox.No)
    def setBrowerPath(self):
        download_path = QtWidgets.QFileDialog.getExistingDirectory(self, "浏览", "C:\\")
        self.lineEdit.setText(download_path)
    def outputWritten(self, text):  # 输出控制台信息
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())