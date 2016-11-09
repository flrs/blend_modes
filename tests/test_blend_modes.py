import cv2
import unittest
from blend_modes.blend_modes import *


class TestBlendModes(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBlendModes, self).__init__(*args, **kwargs)
        self.test_limit = 10 # test fails if max. image color difference is > test_limit/255
        self.img_in = cv2.imread('./orig.png', -1).astype(float)
        self.img_layer = cv2.imread('./layer.png', -1).astype(float)
        self.img_layer_50p = cv2.imread('./layer_50p.png', -1).astype(float)

    def test_addition(self):
        out = soft_light(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./soft_light.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_darken_only(self):
        out = darken_only(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./darken_only.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_difference(self):
        out = difference(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./difference.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_divide(self):
        out = divide(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./divide.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_dodge(self):
        out = dodge(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./dodge.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_grain_extract(self):
        out = grain_extract(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./grain_extract.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_grain_merge(self):
        out = grain_merge(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./grain_merge.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_hard_light(self):
        out = hard_light(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./hard_light.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_lighten_only(self):
        out = lighten_only(self.img_in, self.img_layer, 0.5)
        comp = cv2.imread('./lighten_only.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

    def test_soft_light_50p(self):
        out = soft_light(self.img_in, self.img_layer_50p, 0.8)
        comp = cv2.imread('./soft_light_50p.png', -1).astype(float)
        self.failIf(np.amax(np.absolute(out-comp)) > self.test_limit)

unittest.main()
