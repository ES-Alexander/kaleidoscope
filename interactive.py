#!/usr/bin/env python3

from kaleidoscope import kaleido
from datetime import datetime
import numpy as np
import cv2

class Kaleido:
    def __init__(self, filename, show_input, annotate, window='kaleidoscope'):
        self._image = cv2.imread(filename)
        self._init_variables(show_input, annotate)
        self._init_gui(window)

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
        cv2.createTrackbar(f'r_start [0-359deg]', window,
                           int(self.r_start * 180 / np.pi), 359,
                           self.set_rstart)
        cv2.createTrackbar('r_out [0-1]', window,
                           int(self.r_out / (np.pi / self.N)), 100,
                           self.set_rout)
        cv2.createTrackbar('c_in_x [left-right]', window, self.c_in[1], width,
                           self.set_cix)
        cv2.createTrackbar('c_in_y [top-bottom]', window, self.c_in[0], height,
                           self.set_ciy)

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

    @property
    def processed(self):
        if self._changed:
            self.image = self._image.copy() if self.annotate else self._image
            self._processed = kaleido(self.image, self.N, self.out,
                                      self.r_start, self.r_out, self.c_in,
                                      self.c_out, self.scale, self.annotate)
        return self._processed

    def run(self):
        while (key := (cv2.waitKey(10) & 0xFF)) != ord('q'):
            if key == ord('s'):
                time = datetime.strftime(datetime.now(),
                                         '%Y-%m-%d %H:%M:%S.jpg')
                if self._show_input:
                    cv2.imwrite(f'kaleido-{time}', self._latest)
                cv2.imwrite(f'result-{time}', self._processed)
            self._display()
        print(self)

    def _display(self):
        if self._changed:
            processed = self.processed
            if self._show_input:
                base = self.image
                br = base.shape[0]
                pr = processed.shape[0]
                if br < pr:
                    top = (pr - br) // 2
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
            self._changed = False

        display = self._latest if self._show_input else processed
        cv2.imshow(self._window, display)

    def __repr__(self):
        cls = self.__class__.__name__
        return ('{cls}(N={N}, out={out}, r_start={r_start}, r_out={r_out}, '
                'c_in={c_in}, c_out={c_out}, scale={scale}, '
                'annotate={annotate})').format(cls=cls, **self.__dict__)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', default='test.png',
                        help='path to input image')
    parser.add_argument('-i', '--input', action='store_false',
                        help='flag to turn off input image display')
    parser.add_argument('-a', '--annotate', action='store_false',
                        help='flag to turn off annotation on input image')
    parser.add_argument('-w', '--window', default='kaleidoscope',
                        help='display window name')

    args = parser.parse_args()

    Kaleido(args.filename, args.input, args.annotate, args.window).run()
    cv2.destroyWindow(args.window)

