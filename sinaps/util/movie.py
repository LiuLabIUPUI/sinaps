import numpy as np


def build_movie(shape, nframes=10, nzslices=1, nchannels=1):

    #ImageJ dimension ordering: TZCXY
    shape = (nframes,nzslices,nchannels, *shape)
    movie = np.zeros(shape, dtype=np.float64)

    return movie
