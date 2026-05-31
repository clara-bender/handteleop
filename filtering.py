import numpy as np

class LowPassFilter:
    def __init__(self, alpha, init=None):
        self.alpha = alpha
        self.y = init

    def update(self, x):
        x = np.array(x)

        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1 - self.alpha) * self.y

        return self.y