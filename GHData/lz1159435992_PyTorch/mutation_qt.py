import sys
import gc
from mutation_Window import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import numpy as np
import datetime
import collections
from learning.mnist.model.MutaionOperator import change_weight_
from learning.mnist.model.MutaionOperator import Weight_Shuffing_
from learning.mnist.model.MutaionOperator import Neuron_Activation_Inverse_
from learning.mnist.model.MutaionOperator import Neuron_Effect_Blocking_
from learning.mnist.model.MutaionOperator import Neuron_Switch_
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
    def weight_scaling(self):
        starttime = datetime.datetime.now()
        lines = self.lineEdit.text().split('/')
        filepath1 = ''
        filepath2 = ''
        for i in range(len(lines)):
            if i == 0:
                filepath1 = lines[i]
            else:
                filepath1 = filepath1 + '\\' + lines[i]
        filepath1 = filepath1 + '\\'
        lines = self.lineEdit_2.text().split('/')
        for i in range(len(lines)):
            if i == 0:
                filepath2 = lines[i]
            else:
                filepath2 = filepath2 + '\\' + lines[i]
        filepath2 = filepath2 + '\\'
        #print(self.lineEdit.text())
        #print('**************************************************************************************************************')
        print(filepath1,filepath2)
        change_weight_(filepath1,filepath2)
        #print(self.lineEdit)
        #neural_test('F:\\fault_localization\\test')
        #print(
            #'**************************************************************************************************************')
        # if a == 0:
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
        reply = QMessageBox.information(self, "提示", "Weight_Scaling变异完成,用时"+timeStr, QMessageBox.Yes | QMessageBox.No)
    def weight_shuffing(self):
        starttime = datetime.datetime.now()
        lines = self.lineEdit.text().split('/')
        filepath1 = ''
        filepath2 = ''
        for i in range(len(lines)):
            if i == 0:
                filepath1 = lines[i]
            else:
                filepath1 = filepath1 + '\\' + lines[i]
        filepath1 = filepath1 + '\\'
        lines = self.lineEdit_2.text().split('/')
        for i in range(len(lines)):
            if i == 0:
                filepath2 = lines[i]
            else:
                filepath2 = filepath2 + '\\' + lines[i]
        filepath2 = filepath2 + '\\'
        print(filepath1, filepath2)
        Weight_Shuffing_(filepath1, filepath2)
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
        reply = QMessageBox.information(self, "提示", "Weight_Shuffing变异完成,用时" + timeStr, QMessageBox.Yes | QMessageBox.No)
    def neuron_activation_inverse(self):
        starttime = datetime.datetime.now()
        lines = self.lineEdit.text().split('/')
        filepath1 = ''
        filepath2 = ''
        for i in range(len(lines)):
            if i == 0:
                filepath1 = lines[i]
            else:
                filepath1 = filepath1 + '\\' + lines[i]
        filepath1 = filepath1 + '\\'
        lines = self.lineEdit_2.text().split('/')
        for i in range(len(lines)):
            if i == 0:
                filepath2 = lines[i]
            else:
                filepath2 = filepath2 + '\\' + lines[i]
        filepath2 = filepath2 + '\\'
        print(filepath1, filepath2)
        Neuron_Activation_Inverse_(filepath1, filepath2)
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
        reply = QMessageBox.information(self, "提示", "Neuron_Activation_Inverse变异完成,用时" + timeStr, QMessageBox.Yes | QMessageBox.No)

    def neuron_effect_blocking(self):
        starttime = datetime.datetime.now()
        lines = self.lineEdit.text().split('/')
        filepath1 = ''
        filepath2 = ''
        for i in range(len(lines)):
            if i == 0:
                filepath1 = lines[i]
            else:
                filepath1 = filepath1 + '\\' + lines[i]
        filepath1 = filepath1 + '\\'
        lines = self.lineEdit_2.text().split('/')
        for i in range(len(lines)):
            if i == 0:
                filepath2 = lines[i]
            else:
                filepath2 = filepath2 + '\\' + lines[i]
        filepath2 = filepath2 + '\\'
        print(filepath1, filepath2)
        Neuron_Effect_Blocking_(filepath1, filepath2)
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
        reply = QMessageBox.information(self, "提示", "Neuron_Effect_Blocking变异完成,用时" + timeStr,
                                        QMessageBox.Yes | QMessageBox.No)
    def neuron_switch(self):
        starttime = datetime.datetime.now()
        lines = self.lineEdit.text().split('/')
        filepath1 = ''
        filepath2 = ''
        for i in range(len(lines)):
            if i == 0:
                filepath1 = lines[i]
            else:
                filepath1 = filepath1 + '\\' + lines[i]
        filepath1 = filepath1 + '\\'
        lines = self.lineEdit_2.text().split('/')
        for i in range(len(lines)):
            if i == 0:
                filepath2 = lines[i]
            else:
                filepath2 = filepath2 + '\\' + lines[i]
        filepath2 = filepath2 + '\\'
        print(filepath1, filepath2)
        Neuron_Switch_(filepath1, filepath2)
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
        reply = QMessageBox.information(self, "提示", "Neuron_Switch变异完成,用时" + timeStr,
                                        QMessageBox.Yes | QMessageBox.No)
    def setBrowerPath(self):
        download_path = QtWidgets.QFileDialog.getExistingDirectory(self, "浏览", "C:\\")
        self.lineEdit.setText(download_path)
    def setBrowerPath_2(self):
        download_path = QtWidgets.QFileDialog.getExistingDirectory(self, "浏览", "C:\\")
        self.lineEdit_2.setText(download_path)
    def outputWritten(self, text):  # 输出控制台信息
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()
if __name__ == '__main__':
    #change_weight_(r'C:\Users\LZ\PycharmProjects\PyTorch\learning\mnist\model\\',r'F:\fault_localization\test')
    app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QMainWindow()
    window = mywindow()

    window.show()
    # child = ChildWindow()
    # child_ui = Ui_Dialog()
    # child_ui.setupUi(child)
    # child_ui = Ui_Dialog()
    # child_ui.setupUi(child)
    sys.exit(app.exec_())