import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
import pims

from sinaps.util import *
from sinaps.track import *
from sinaps.brownian import *
from sinaps.plot import *
from sinaps.noise import *
from cellquantifier.smt import *
from cellquantifier.qmath import *

settings = {

	#io parameters
	'output_dir': '/home/clayton/Desktop/temp/',

	#spatial parameters (observation volume)
	'n_dim': 2,
	'x_size': 100,
	'y_size': 100,
	'z_size': 100,

	#temporal parameters
	'acq_time': .5, #s
	'exposure_time': 1, #s
	'frame_rate': 200, #fps

	#emitter parameters
	'deg_freedom': 2, #degrees of freedom
	'lat_dim': 2, #lattice dimension
	'nparticles': 100,
	'ex_wavelength': 488, #nm
	'em_wavelength': 520, #nm
	'photon_rate': 50, #photons per second
	'particle_radius': 10e-9, #m - quantum dot radius
	'particle mass': 1e-10, #kg

	#camera parameters
	'pixel_size': 3.75, #um
	'quant_eff': .69, #elec/photon
	'sensitivity': 5.88,
	'sigma_dark': 0, #elec
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
lattice = gen_lattice(nparticles=settings.NPARTICLES,
					  ndim=settings.LAT_DIM,
					  show_lattice=False)

#Run the simulation
movie = Movie(settings)
movie.build_brute_traj(origins=lattice, dcoeff=.05)
movie.add_photon_stats()
movie.simulate()
movie.add_noise()
movie.save(filename='200402_77Sim-0-moving.tif')

# frames = pims.open(settings.OUTPUT_DIR + '/200402_77Sim-0-moving.tif')
# df = det_fit_track(frames, blob_thres=.1)
# df.to_csv(settings.OUTPUT_DIR + '/200402_77Sim-0-moving.csv')
#
# df = pd.read_csv(settings.OUTPUT_DIR + '/200402_77Sim-0-moving.csv')
# all_msd, all_std = get_msd(df)
# mean_msd = np.mean(all_msd, axis=1)
# dcoeff = fit_brownian(mean_msd.index, mean_msd)
# t = mean_msd.index
# # plt.plot(t, 2*dcoeff*t)
# plt.plot(t, mean_msd)
# plt.errorbar(t, mean_msd, yerr=all_std)
# plt.show()
