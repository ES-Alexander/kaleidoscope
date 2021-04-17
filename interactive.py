#!/usr/bin/env python3

from kaleidoscope import kaleido
from datetime import datetime
import numpy as np
import cv2
try:
    from pcv.vidIO import Camera, UserQuit
    from sys import exit
except ImportError:
    print('failed to import pythonic-cv (pcv) - live mode disabled')

class Kaleido:
    def __init__(self, filename, show_input, annotate, window='kaleidoscope',
                 live=False):
        self._path  = filename
        if not live:
            self.__image = cv2.imread(filename)
            self._live = False
        else:
            self._live = True
            self.__cam = filename
        self._init_variables(show_input, annotate)
        self._init_gui(window)

    @property
    def _image(self):
        if self._live:
            try:
                return cv2.flip(next(self.__cam)[1], 1)
            except UserQuit:
                exit(0) # camera stream over by choice of user - stop updating
        return self.__image

    def _init_variables(self, show_input, annotate):
        self.annotate = annotate and show_input # only annotate if visible
        self.N        = 10
        self.out      = 'full'
        self.r_start  = 0
        self.r_out    = 0
        self.c_in     = np.array(self._image.shape[:2]) // 2
        self.c_out    = None
        self.scale    = 1
        self._changed = True
        self._show_input = show_input

    def _init_gui(self, window):
        self._window = window
        height, width = self._image.shape[:2]
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('scale [1-200%]', window, int(self.scale * 100),
                           199, self.set_scale)
        cv2.createTrackbar('N [1-20]', window, self.N, 19, self.set_N)
        
        cv2.createTrackbar('r_out [0-1]', window,
                           int(self.r_out / (np.pi / self.N)), 100,
                           self.set_rout)
        if self._show_input:
            print('click and drag on input image to set center and direction')
            self._moved = False
            cv2.setMouseCallback(window, self.mouse_event)
        else:
            cv2.createTrackbar(f'r_start [0-359deg]', window,
                               int(self.r_start * 180 / np.pi), 359,
                               self.set_rstart)
            cv2.createTrackbar('c_in_x [left-right]', window, self.c_in[1],
                               width, self.set_cix)
            cv2.createTrackbar('c_in_y [top-bottom]', window, self.c_in[0],
                               height, self.set_ciy)

    def _changer(func):
        def wrapped(self, *args):
            self._changed = True
            return func(self, *args)
        return wrapped

    @_changer
    def set_scale(self, scale):
        self.scale = (scale + 1) / 100

    @_changer
    def set_N(self, N):
        self.N = N + 1

    @_changer
    def set_rstart(self, r_start):
        self.r_start = r_start * np.pi / 180

    @_changer
    def set_rout(self, r_out):
        self.r_out = r_out * (np.pi / self.N) / 100

    @_changer
    def set_cix(self, c_in_x):
        self.c_in[1] = c_in_x

    @_changer
    def set_ciy(self, c_in_y):
        self.c_in[0] = c_in_y

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__x = x
            self.__y = y - self._offset
            self._backup = self._latest.copy()
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                x, y = self._arrow_end(x, y-self._offset)

                self._latest = self._backup.copy()
                cv2.arrowedLine(self._latest,
                                (self.__x, self.__y + self._offset),
                                (int(x), int(y + self._offset)), (255,255,255))
                self._moved = True
        elif event == cv2.EVENT_LBUTTONUP:
            if self._moved:
                self.set_cix(self.__x)
                self.set_ciy(self.__y)
                self.r_start = np.arctan2((y-self._offset)-self.__y,
                                          x-self.__x) - np.pi / (2 * self.N)
                self._moved = False

    def _arrow_end(self, x, y):
        angle = np.arctan2(y-self.__y, x-self.__x)
        rows, cols = self._image.shape[:2]
        # angle range towards each side, for a point inside the image
        bottom_right, top_right, top_left, bottom_left = np.arctan2(
            [rows-y, -y,     -y, rows-y],
            [cols-x, cols-x, -x, -x    ]
        )
        # NOTE:
        #     0 <= br <=  pi/2
        #     0 >= tr >= -pi/2
        # -pi/2 >= tl >= -pi
        #  pi/2 <= bl <=  pi

        if top_left <= angle <= top_right:
            # go to top line
            return (self.__x + np.tan(np.pi/2 + angle) * self.__y, 0)
        if bottom_right <= angle <= bottom_left:
            # go to bottom line
            return (self.__x + np.tan(np.pi/2 - angle) * (rows - self.__y),
                    rows)
        if (0 <= angle <= bottom_right) or (top_right <= angle <= 0):
            # go to right side
            return (cols, self.__y + np.tan(angle) * (cols - self.__x))
        # else:
        # go to left side
        return (0, self.__y + np.tan(np.pi - angle) * self.__x)

    @property
    def processed(self):
        if self._changed or self._live:
            self.image = self._image.copy() if self.annotate else self._image
            self._processed = kaleido(self.image, self.N, self.out,
                                      self.r_start, self.r_out, self.c_in,
                                      self.c_out, self.scale, self.annotate)
        return self._processed

    def run(self):
        print("Press 'q' or ESC to quit, or 's' to save the current result.")
        while (key := (cv2.waitKey(1) & 0xFF)) not in (ord('q'), 27):
            if key == ord('s'):
                time = datetime.strftime(datetime.now(),
                                         '%Y-%m-%d_%H-%M-%S.jpg')
                if self._show_input:
                    cv2.imwrite(f'kaleido_{time}', self._latest)
                cv2.imwrite(f'result_{time}', self._processed)
            self._display()
        print(self)

    def _display(self):
        if self._changed or (self._live and
                             ((not self._show_input) or
                              (self._show_input and not self._moved))):
            processed = self.processed
            if self._show_input:
                self._offset = 0
                base = self.image
                br = base.shape[0]
                pr = processed.shape[0]
                if br < pr:
                    self._offset = top = (pr - br) // 2
                    bottom = pr - top - br
                    base = cv2.copyMakeBorder(base, top, bottom, 0, 0,
                                              cv2.BORDER_CONSTANT, value=[0]*3)
                elif pr < br:
                    top = (br - pr) // 2
                    bottom = br - top - pr
                    processed = cv2.copyMakeBorder(processed, top, bottom, 0,
                                                   0, cv2.BORDER_CONSTANT,
                                                   value=[0]*3)
                self._latest = cv2.hconcat((base, processed))

        if self._changed or self._live or (self._show_input and self._moved):
            display = self._latest if self._show_input else self._processed
            cv2.imshow(self._window, display)
            self._changed = False

    def __repr__(self):
        cls = self.__class__.__name__
        return ('{cls}(image={_path}, N={N}, out={out}, r_start={r_start}, '
                'r_out={r_out}, c_in={c_in}, c_out={c_out}, scale={scale}, '
                'annotate={annotate})').format(cls=cls, **self.__dict__)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', default='test.png',
                        help='path to input image, or camera ID')
    parser.add_argument('-i', '--input', action='store_false',
                        help='flag to turn off input image display')
    parser.add_argument('-a', '--annotate', action='store_false',
                        help='flag to turn off annotation on input image')
    parser.add_argument('-w', '--window', default='kaleidoscope',
                        help='display window name')
    parser.add_argument('-l', '--live', action='store_true',
                        help='run live on your webcam instead of an image')

    args = parser.parse_args()

    if args.live:
        with Camera(int(args.filename), display=args.window) as cam:
            print('here')
            Kaleido(iter(cam), args.input, args.annotate, args.window,
                    args.live).run()
    else:
        Kaleido(args.filename, args.input, args.annotate, args.window).run()
        cv2.destroyWindow(args.window)

