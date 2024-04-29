import collections
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, OrderedDict, Dict
from dataclasses import dataclass
import cv2
import numpy as np
import pyqtgraph as pg
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QVBoxLayout, \
    QHBoxLayout, QGroupBox, QLabel, QComboBox, QInputDialog, QFileDialog, QLineEdit, \
    QRadioButton, QButtonGroup, QScrollArea


@dataclass
class ImageTypeCfg:
    path: str
    pattern: str
    action: str
    pivot: bool = False


def crop_center(img, cropy, cropx):
    y, x = img.shape[:2]
    cropy, cropx = min(cropy, img.shape[0]), min(cropx, img.shape[1])

    sx = x//2-(cropx//2)
    sx -= sx % 2
    sy = y//2-(cropy//2)
    sy -= sy % 2
    return img[sy:sy+cropy, sx:sx+cropx]


def auto_gamma_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = np.log(mid * 255) / np.log(mean)
    # print(gamma)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    corrected = cv2.LUT(img, table)
    return cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)


def auto_gamma_gray(img):
    max_val = 2**16 - 1 if img.dtype == np.uint16 else 2**8 - 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid = 0.45
    mean = np.mean(gray)
    gamma = np.log(mid * max_val) / np.log(mean)
    corrected = np.power(img, gamma).clip(0, max_val).astype(img.dtype)
    return corrected


class ImageTypeView(QGroupBox):
    def __init__(self, parent, name: str, cfg: ImageTypeCfg):
        super().__init__(parent=parent, title=name)
        self.src_dir = Path(cfg.path)
        self.files: Dict[str, Path] = []
        self.convert = None
        self.name = name
        # UI
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        # view box
        self.vb = pg.ViewBox(lockAspect=True)
        gv = pg.GraphicsView(useOpenGL=False)
        gv.setCentralItem(self.vb)
        self.it = pg.ImageItem()
        self.it.setImage()
        self.vb.addItem(self.it)
        layout.addWidget(gv)
        if str(self.src_dir) == 'mock':
            self.files = None
        else:
            self.files = {}
            for im_path in self.src_dir.glob(cfg.pattern):
                self.files[im_path.name.replace(cfg.pattern.replace('*', ''), '')] = im_path
        self.convert = []
        if 'mono' in cfg.action:
            self.convert.append(lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        if 'rgb' in cfg.action:
            self.convert.append(lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if 'bgr' in cfg.action:
            self.convert.append(lambda im: im)
        if 'rggb' in cfg.action:
            self.convert.append(lambda im: cv2.demosaicing(im, cv2.COLOR_BAYER_BG2RGB))
        if 'gamma' in cfg.action:
            self.convert.append(auto_gamma_gray)
        if 'crop' in cfg.action:
            self.convert.append(lambda im: crop_center(im, 2048, 2048))

    def display_image(self, im_base_name):
        if self.files is None:
            self.it.setImage(np.zeros((500, 500, 3), dtype=np.uint8))
            self.vb.autoRange()
            return
        if im_base_name in self.files:
            im = cv2.imread(str(self.files[im_base_name]), cv2.IMREAD_UNCHANGED)
            for conversion in self.convert:
                im = conversion(im)
            if im.dtype == np.uint16:
                im = (im // 256).astype(np.uint8)
            self.it.setImage(np.rot90(im, 3), autoLevels=False, levelSamples=255)
            self.vb.autoRange()
        else:
            self.it.setImage(np.zeros((500, 500, 3), dtype=np.uint8))
            # self.vb.autoRange()
            print(f'{self.name}: {im_base_name} not found in {list(self.files.values())}')


class MainWin(QWidget):
    def __init__(self, cfg_path):
        super().__init__()
        with cfg_path.open('r') as f:
            cfg = yaml.safe_load(f)
            if 'cols' in cfg:
                divisor = cfg['cols']
            elif 'rows' in cfg:
                divisor = cfg['rows'] // len(cfg['inputs']) + cfg['rows'] % len(cfg['inputs'])
        self.inputs: OrderedDict[str, ImageTypeView] = collections.OrderedDict()
        self.setGeometry(100, 100, 1200, 800)
        self.multi_layout = QGridLayout()
        self.setLayout(self.multi_layout)
        self.base_names = None
        self.pivot: ImageTypeView = None
        for idx, (im_name, im_params) in enumerate(cfg['inputs'].items()):
            im_cfg = ImageTypeCfg(**im_params)
            self.inputs[im_name] = ImageTypeView(self, im_name, im_cfg)
            self.multi_layout.addWidget(self.inputs[im_name], idx//divisor, idx % divisor)
            if self.base_names is None and im_cfg.pivot:
                self.pivot = self.inputs[im_name]
        if self.pivot is None:
            self.pivot = list(self.inputs.values())[0]
        for im_type in self.inputs.values():
            if im_type is not self.pivot and im_type.files is not None:
                # self.pivot.vb.setXLink(im_type.vb)
                # self.pivot.vb.setYLink(im_type.vb)
                im_type.vb.setXLink(self.pivot.vb)
                im_type.vb.setYLink(self.pivot.vb)
        self.base_names = list(self.pivot.files.keys())
        self.ind = 0
        self.set_image(self.ind)
        self.show()

    def set_image(self, ind):
        for im_name, im_view in self.inputs.items():
            im_view.display_image(self.base_names[ind])
        self.setWindowTitle(self.base_names[ind])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right and self.ind < len(self.base_names) - 1:
            self.ind += 1
            self.set_image(self.ind)
        elif event.key() == Qt.Key_Left and self.ind > 0:
            self.ind -= 1
            self.set_image(self.ind)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg_file', help='data set parameters file path')
    args = parser.parse_args()

    app = QApplication([])
    cfg_path = Path(args.cfg_file)
    assert cfg_path.exists(), f'failed to find input file {args.cfg_file} '
    gui = MainWin(cfg_path)
    app.exec_()
