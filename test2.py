from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from sinaps.brownian import build_traj2
from sinaps.util import *
from sinaps.photon import *
from trackpy import imsd
import matplotlib.pyplot as plt

#
#~~~~~~~params~~~~~~~~~~
#

path = '/home/cwseitz/Desktop/test.tif'
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
#~~~~~~~build the movie~~~~~~~~~~
#

traj_df = add_photon_stats(traj_df, add_noise=False); print(traj_df)
movie = add_psfs_batch(movie, traj_df)
movie = add_noise_batch(movie, sigma_dark=0.1)
movie = normalize_by_frame(movie)
movie = img_as_ubyte(movie)
imsave(path, movie)
