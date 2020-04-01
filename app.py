import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
from movie import Movie
from settings import Settings
from util import *
from brownian import *
from plot import *

settings = {

    #io parameters
    'output_dir': '/home/clayton/Desktop/temp/',

    #spatial parameters (observation volume)
    'n_dim': 2,
    'x_size': 100,
    'y_size': 100,
    'z_size': 100,

    #temporal parameters
    'acq_time': 10, #s
    'exposure_time': 1, #s
    'frame_rate': 10, #fps

    #emitter parameters
    'deg_freedom': 2, #degrees of freedom
    'lat_dim': 2, #lattice dimension
    'nparticles': 49,
    'ex_wavelength': 488, #nm
    'em_wavelength': 520, #nm
    'photon_rate': 50, #photons per second
    'particle_radius': 10e-9, #m - quantum dot radius
    'particle mass': 1e-10, #kg

    #camera parameters
    'pixel_size': 3.75, #um
    'quant_eff': .69, #elec/photon
    'sensitivity': 5.88,
    'sigma_dark': 2.29, #elec
    'bit_depth': 8,
    'baseline': 10, #adu
    'dynamic_range': 1000, #max photons per pixel

    #microscopy parameters
    'num_aperture': 1.22,
    'refr_index': 1.333,
    'pinhole_rad': .55,
    'pinhole_shape': 'round',

    #environmental parameters
    'env_temp': 310, #kelvin - 37C
    'env_visc': .002, #Pa*s - oil
}


settings = Settings(settings)

#Generate Origins
# lattice = gen_lattice(nparticles=settings.NPARTICLES,
#                       ndim=settings.LAT_DIM,
#                       show_lattice=False)
#
# #Build the trajectories
# corr_mat_3d = get_corr_2D(settings.NPARTICLES,
#                        settings.NFRAMES)
#
# std = np.ones(settings.NPARTICLES)

# corr_mat_3d=None
# std=None


#Run the simulation
movie = Movie(settings)
movie.build_traj(corr_mat_3d=None, std=None, origins=None)
particles = pd.unique(movie.traj_df['particle'])
im = tp.imsd(movie.traj_df, mpp=1, fps=1)
mean_msd = np.mean(im, axis=1)
n = int(len(mean_msd)/5)
mean_msd = mean_msd[:n]
plt.plot(mean_msd)
plt.show()

# show_force_comp(movie.traj_df)
# show_traj_batch(movie.traj_df,ndim=2)
# movie.add_photon_stats()
# movie.simulate()
# movie.add_noise()
# movie.save(corr_mat_3d)
