from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QVBoxLayout, \
    QHBoxLayout, QSlider, QGroupBox, QLabel, QShortcut
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QMouseEvent
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Dict
import numpy as np
from PIL import Image
import os.path
from shutil import rmtree
import cv2
import matplotlib.pyplot as plt


class MyViewBox(pg.ViewBox):
    def __init__(self, draw_function):
        self.draw_func = draw_function
        super().__init__()

    def mousePressEvent(self, QMouseEvent):
        self.draw_func(QMouseEvent.pos())
        print(QMouseEvent.pos())



def comb_view(name, im_layout, i, draw_function = None) -> (pg.ImageItem, pg.ViewBox):
    if draw_function is not None:
        vb = MyViewBox(draw_function=draw_function)
    else:
        vb = pg.ViewBox(lockAspect=True)
    gv = pg.GraphicsView(useOpenGL=False)
    gv.setCentralItem(vb)
    it = pg.ImageItem()
    it.setImage()
    vb.addItem(it)
    cam_group = QGroupBox(name)
    cam_layout = QVBoxLayout()
    cam_layout.addWidget(gv)
    cam_group.setLayout(cam_layout)
    im_layout.addWidget(cam_group, i % 2, i // 2)
    return it, vb


class Tagger(QWidget):
    def __init__(self, wins_generator, diff_id, segment_id, output_folder, params):
        super().__init__()
        self.data_generator = self.detection_generator(wins_generator)
        self.current_container = None
        self.current_detection = None
        self.output_folder = output_folder
        self.example_idx = 0
        self.ret = None
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        side_layout = QVBoxLayout()
        main_layout.addLayout(side_layout)
        im_layout = QGridLayout()
        main_layout.addLayout(im_layout)
        self.clips: Dict[str, pg.ImageItem] = {}
        self.boxes: Dict[str, pg.ViewBox] = {}
        self.sliders: Dict[str, QSlider] = {}
        self.labels: Dict[str, QLabel] = {}
        self.defaults: Dict[str, int] = {'diff_id': diff_id}
        for i, im_type in enumerate(['current', 'reference', 'mask', 'diff']):
            if im_type == 'mask':
                it, vb = comb_view(im_type, im_layout, i, draw_function=self.draw)
            else:
                it, vb = comb_view(im_type, im_layout, i)
            self.clips[im_type] = it
            self.boxes[im_type] = vb
            if i > 0:
                self.boxes[im_type].setXLink(self.boxes['current'])
                self.boxes[im_type].setYLink(self.boxes['current'])

        # threshold layout
        grid_row = 0
        th_layout = QGridLayout()
        self.debug = QPushButton('debug')
        self.debug.clicked.connect(self.refresh)
        self.debug.setShortcut(QKeySequence(Qt.Key_P))
        self.debug.setCheckable(True)
        side_layout.addWidget(self.debug)

        self.auto_level_btn = QPushButton('auto level')
        self.auto_level_btn.clicked.connect(self.auto_th)
        self.auto_level_btn.setShortcut(QKeySequence(Qt.Key_L))
        side_layout.addWidget(self.auto_level_btn)

        self.mask_add_btn = QPushButton('add to mask')
        self.mask_add_btn.clicked.connect(self.add_mask)
        self.mask_add_btn.setCheckable(True)
        self.mask_add_btn.setShortcut(QKeySequence(Qt.Key_M))
        side_layout.addWidget(self.mask_add_btn)
        self.mask = None
        self.curr_mask = None

        self.my_slider('segment TH', min_val=5, max_val=120, val=params['window']['segmentation_th'], layout=th_layout, grid_row=grid_row)
        grid_row += 1
        self.my_slider('edge TH', min_val=1, max_val=40, val=params['edge_th'], layout=th_layout, grid_row=grid_row, interval=0.5)
        grid_row += 1
        self.my_slider('dilation', min_val=0, max_val=10, val=0, layout=th_layout, grid_row=grid_row, is_default=True)
        grid_row += 1
        self.my_slider('dx', min_val=-60, max_val=60, val=0, layout=th_layout, grid_row=grid_row, is_default=True)
        grid_row += 1
        self.my_slider('dy', min_val=-60, max_val=60, val=0, layout=th_layout, grid_row=grid_row, is_default=True)
        grid_row += 1
        self.my_slider('reg dx', min_val=-60, max_val=60, val=0, layout=th_layout, grid_row=grid_row, is_default=True)
        grid_row += 1
        self.my_slider('reg dy', min_val=-60, max_val=60, val=0, layout=th_layout, grid_row=grid_row, is_default=True)
        grid_row += 1
        self.my_slider('segment id', min_val=1, max_val=segment_id - 1,
                       val=segment_id - 1, layout=th_layout, grid_row=grid_row, is_default=False)
        grid_row += 1
        side_layout.addLayout(th_layout)

        next_btn = QPushButton('next')
        next_btn.setShortcut(QKeySequence(Qt.Key_Space))
        side_layout.addWidget(next_btn)
        next_btn.clicked.connect(self.step_next)

        save_btn = QPushButton('save')
        save_btn.setShortcut(QKeySequence(Qt.Key_E))
        side_layout.addWidget(save_btn)
        save_btn.clicked.connect(self.save)
        self.step_next()
        self.show()

    def my_slider(self, name, min_val, max_val, val, layout, grid_row, is_default=False, interval=1):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimumWidth(200)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(val)
        slider.valueChanged.connect(self.refresh)
        slider.setSingleStep(interval)
        self.labels[name] = QLabel('{}'.format(slider.value()))
        layout.addWidget(QLabel(name), grid_row, 0)
        layout.addWidget(self.labels[name], grid_row, 1)
        layout.addWidget(slider, grid_row, 2)
        if is_default:
            self.defaults[name] = val
        self.sliders[name] = slider

    def draw(self, point):
        if self.curr_mask is not None:
            print(point)
            # self.curr_mask[int(point.x()), int(point.y())] = not self.curr_mask[int(point.x()), int(point.y())]
            # res_cnt, _ = cv2.findContours(self.curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # ann = self.current_ann.copy()
            # cv2.drawContours(ann, res_cnt, contourIdx=-1, color=(255, 0, 0), thickness=1)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.sliders['segment TH'].setValue(self.sliders['segment TH'].value() + 1)
        elif event.key() == Qt.Key_Left:
            self.sliders['segment TH'].setValue(self.sliders['segment TH'].value() - 1)
        elif event.key() == Qt.Key_Up:
            self.sliders['edge TH'].setValue(self.sliders['edge TH'].value() + 1)
        elif event.key() == Qt.Key_Down:
            self.sliders['edge TH'].setValue(self.sliders['edge TH'].value() - 1)
        elif event.key() == Qt.Key_A:
            self.sliders['reg dx'].setValue(self.sliders['reg dx'].value() + 1)
        elif event.key() == Qt.Key_D:
            self.sliders['reg dx'].setValue(self.sliders['reg dx'].value() - 1)
        elif event.key() == Qt.Key_W:
            self.sliders['reg dy'].setValue(self.sliders['reg dy'].value() + 1)
        elif event.key() == Qt.Key_S:
            self.sliders['reg dy'].setValue(self.sliders['reg dy'].value() - 1)
        else:
            QWidget.keyPressEvent(self, event)

    def display_detection(self, win_container, detection, new=False):
        x, y, w, h = detection
        self.ret = views(win_container,
                         x=x + self.sliders['dx'].value(),
                         y=y + self.sliders['dy'].value(),
                         w=128,
                         th=self.sliders['segment TH'].value(),
                         debug=self.debug.isChecked(),
                         edge_th=self.sliders['edge TH'].value() / 2,
                         diff_win_id=self.defaults['diff_id'],
                         seg_win_id=self.sliders['segment id'].value(),
                         dy=self.sliders['reg dy'].value(),
                         dx=self.sliders['reg dx'].value(),
                         dilation=self.sliders['dilation'].value())

        if self.ret:
            diff, self.current_ann, self.curr_mask, reference, _, _ = self.ret
            if self.mask is not None:
                self.curr_mask = np.bitwise_or(self.curr_mask, self.mask)
            res_cnt, _ = cv2.findContours(self.curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ann = self.current_ann.copy()
            cv2.drawContours(ann, res_cnt, contourIdx=-1, color=(255, 0, 0), thickness=0)
            self.clips['current'].setImage(np.rot90(ann, 3))
            self.clips['reference'].setImage(np.rot90(reference, 3))
            self.clips['mask'].setImage(np.rot90(self.curr_mask, 3))
            self.clips['diff'].setImage(np.rot90(diff, 3))
        if new:
            for box in self.boxes.values():
                box.autoRange()
            # plt.hist(diff.flatten(), 256)
            # plt.show()



    def auto_th(self):
        diff, _, _, _, edge_ranks, prev_ranks = self.ret
        print('percentile', np.percentile(diff, [90]))
        print('mean', np.mean(diff))
        print('std', np.std(diff))
        self.sliders['segment TH'].setValue(np.mean(diff) + (3 * np.std(diff)))
        self.refresh()
        if len(edge_ranks) > 0:
            e_th = min(edge_ranks) + min(edge_ranks) / 10
            self.sliders['edge TH'].setValue(e_th - (e_th % 0.5))
            self.refresh()


    def refresh(self):
        for slider_name, slider in self.sliders.items():
            if slider_name=='edge TH':
                self.labels[slider_name].setText('{}'.format(slider.value() / 2))
            else:
                self.labels[slider_name].setText('{}'.format(slider.value()))
        if self.current_container is not None:
            self.display_detection(self.current_container, self.current_detection)

    def step_next(self):
        self.mask = None
        self.mask_add_btn.setChecked(False)
        self.current_container, self.current_detection = next(self.data_generator)
        for slider_name, slider in self.sliders.items():
            if slider_name in self.defaults:
                slider.setValue(self.defaults[slider_name])
        self.display_detection(self.current_container, self.current_detection, new=True)

    def add_mask(self):
        if self.mask_add_btn.isChecked():
            self.mask = self.curr_mask
        else:
            self.mask = None
        self.refresh()

    def save(self):
        x, y, w, h = self.current_detection
        ret = views(win_cont=self.current_container,
                    x=x + self.sliders['dx'].value(),
                    y=y + self.sliders['dy'].value(),
                    w=128,
                    th=self.sliders['segment TH'].value(),
                    debug=False,
                    edge_th=self.sliders['edge TH'].value() / 2,
                    diff_win_id=self.defaults['diff_id'],
                    seg_win_id=self.sliders['segment id'].value(),
                    dy=self.sliders['reg dy'].value(),
                    dx=self.sliders['reg dx'].value(),
                    dilation=self.sliders['dilation'].value(),
                    export_data=True)

        if ret is not None and self.output_folder is not None:
            patchs, mask = ret
            if self.mask is not None:
                mask = np.bitwise_or(mask, self.mask)
            exp_folder = os.path.join(self.output_folder, '{exp:04d}'.format(exp=self.example_idx))
            if os.path.isdir(exp_folder):
                rmtree(exp_folder)
            os.makedirs(exp_folder)

            for idx, patch in enumerate(patchs):
                if patch is None:
                    break
                im = Image.fromarray(patchs[idx])
                name = '{id:02d}.png'.format(id=idx)
            #     name = '{exp:04d}_image.png'.format(exp=self.example_idx)
                im.save(os.path.join(exp_folder, name))

            im = Image.fromarray(mask)
            mask_name = 'mask.png'
            im.save(os.path.join(exp_folder, mask_name))
            # print('saved {} patches and mask for example {}'.format(idx, self.example_idx))
            self.example_idx += 1
            self.step_next()

    @staticmethod
    def detection_generator(wins_generator):
        for detected_win in wins_generator:
            for win_container in detected_win.flatten():
                # print('win {}, {}'.format(win_container.i, win_container.j))
                # if 5 < win_container.i < 13 and win_container.j < 17:
                #     continue
                for detection in tag_diff(win_container):
                    yield win_container, detection


def run_frames(path, params) -> WinContainer:
    video = video_wrapper(path, step=params['frame_sample_step'])
    frame0 = next(video)
    params = update_params(frame0, params)
    handler = FrameHandler(frame0, params, None)
    for frame in video:
        handler.add_frame(frame)
        if handler.do_detection:
            yield handler.win_containers


if __name__ == '__main__':
    parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg_file', help='data set parameters file path')
    parser.add_argument('--seq_frames', nargs='+', help='first image path')
    parser.add_argument('-f', '--video_path', help='images folder')
    parser.add_argument('-fd', '--frame_debug', action='store_true', help='add frame debug prints/plots', default=False)
    parser.add_argument('-wd', '--window_debug', action='store_true', help='add window debug prints/plots', default=False)
    parser.add_argument('-o', '--output_path', default=None)
    args = parser.parse_args()

    params = read_parameters_file(args.cfg_file)
    params['frame']['debug'] = args.frame_debug
    params['window']['debug'] = args.window_debug
    params.update({'job_name': None,
                   'add_hog': False,
                   'save_segmentation': False,
                   'segmentation_min_sample_step': 80})

    detection_accumulate_size = np.ceil(
        params['motion_detection_min_sample_step'] / params['frame_sample_step']).astype(int)
    segmentation_accumulate_size = np.ceil(
        params['segmentation_min_sample_step'] / params['frame_sample_step']).astype(int)
    wins_generator = run_frames(args.video_path, params)
    app = QApplication([])
    gui = Tagger(wins_generator, detection_accumulate_size, segmentation_accumulate_size, args.output_path, params)
    app.exec_()
