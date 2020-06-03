from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from copy import deepcopy
from sinaps.brownian import build_traj2
from sinaps.util import *
from sinaps.photon import *
from trackpy import imsd
import matplotlib.pyplot as plt

#
#~~~~~~~params~~~~~~~~~~
#

path1 = '/home/cwseitz/Desktop/test.tif'
path2 = '/home/cwseitz/Desktop/test2.tif'
shape = (100,100)
nparticles=16
nframes=10
lattice_size=20

#
#~~~~~~~generate trajectories~~~~~~~~~~
#

movie = build_movie(shape, nframes=nframes)
origins = gen_lattice(nparticles=nparticles, show_lattice=False, ndim=2)
origins = lattice_size*origins
traj_df = build_traj2(nparticles=nparticles, nframes=nframes, origins=origins, dcoeff=.1)

#
#~~~~~~~build noisy and noiseless movies~~~~~~~~~~
#

traj_df = add_photon_stats(traj_df, add_noise=False)
no_noise = add_psfs_batch(movie, traj_df)

noise = deepcopy(no_noise)
noise = add_noise_batch(noise, sigma_dark=0.1)

no_noise = normalize_by_frame(no_noise)
no_noise = img_as_ubyte(no_noise)

noise = normalize_by_frame(noise)
noise = img_as_ubyte(noise)

imsave(path1, noise)
imsave(path2, no_noise)
