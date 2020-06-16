#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np

class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        if not np.isscalar(x):
            x = np.asarray(x)
        return self.coef * x

class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)

class GridNormalizer(RescaleNormalizer):
    def __init__(self, gridsize):
        RescaleNormalizer.__init__(self, 1.0 / gridsize)

class IdentityNormalizer(BaseNormalizer):
    def __call__(self, x):
        return x