import numpy as np

implemented_errors = ['mse', 'mad', 'sad', 'ssd']

def compute_error(x1, x2, mode):
    if mode == "mse":
        return mse(x1, x2)
    if mode == "mad":
        return mad(x1, x2)
    if mode == "sad":
        return sad(x1, x2)
    if mode == "ssd":
        return ssd(x1, x2)
    if mode == "psnr":
        return psnr(x1, x2)
    if mode == "itf":
        return itf(x1, x2)
    raise NotImplemented("Choose one from the list: {}".format(implemented_errors))

def mse(x1, x2):
    pass

def mad(x1, x2):
    pass

def sad(x1, x2):
    return sum(sum(abs(x1-x2)**2))

def psnr(x1,x2):
    pass

# Interframe Transformation Fidelity
def itf(x1, x2):
    pass


def ssd(x1,x2):
    pass