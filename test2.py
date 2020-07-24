from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from sinaps.brownian import get_trajectories
from cellquantifier.smt import get_d_values
from cellquantifier.plot import plot_phys_1
from sinaps.util import *
from sinaps.photon import *
import trackpy as tp
from copy import deepcopy
import matplotlib.pyplot as plt

#
#~~~~~~~params~~~~~~~~~~
#

path1 = '/home/cwseitz/Desktop/test.tif'
path2 = '/home/cwseitz/Desktop/test2.tif'
shape = (100,100)
nparticles=49
nframes=100
lattice_size=30
dcoeff=.01
pixel_size=.001 #mpp
frame_rate=1 #fps
divide_num=5
sigma_dark=0.1

#
#~~~~~~~generate trajectories~~~~~~~~~~
#

movie = build_movie(shape, nframes=nframes)
origins = gen_lattice(nparticles=nparticles, show_lattice=False, ndim=2)
origins = lattice_size*origins

df = get_trajectories(nparticles=nparticles,
					   nframes=nframes,
					   dcoeff=dcoeff,
					   method='individual',
					   origins=origins)

#
#~~~~~~~individual msd~~~~~~~~~~
#

im = tp.imsd(df, mpp=pixel_size, fps=frame_rate)
df = get_d_values(df, im, divide_num=divide_num)
df = df.assign(tmp='tmp')
plot_phys_1(df,
			cat_col='tmp',
            pixel_size=pixel_size,
            frame_rate=frame_rate,
            divide_num=divide_num)
plt.show()

#
#~~~~~~~build noisy and noiseless movies~~~~~~~~~~
#

df = add_photon_stats(df, add_noise=False)
no_noise = add_psfs_batch(movie, df)

noise = deepcopy(no_noise)
noise = add_noise_batch(noise, sigma_dark=sigma_dark)

no_noise = normalize_by_frame(no_noise)
no_noise = img_as_ubyte(no_noise)

noise = normalize_by_frame(noise)
noise = img_as_ubyte(noise)

imsave(path1, noise)
imsave(path2, no_noise)
