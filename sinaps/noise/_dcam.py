from skimage.filters import gaussian
from skimage.util import img_as_ubyte
import psf
import numpy as np

def add_noise(frame,
			  quant_eff=1,
			  sigma_dark=10,
			  bit_depth=8,
			  sensitivity=1,
			  baseline=0,
			  dyn_range=1000):

	"""Model real camera noise: shot, dark, and discretization

	Parameters
	----------
	"""


	# Convert to electrons
	electrons = quant_eff*frame
	# Add dark noise
	electrons = np.random.normal(scale=sigma_dark, \
							  size=electrons.shape) + electrons
	# Convert to ADU and add baseline
	max_adu     = np.int(2**bit_depth - 1)
	adu         = (electrons * sensitivity).astype(np.int)
	adu += baseline
	adu[adu > max_adu] = max_adu # models pixel saturation
	electrons = electrons/electrons.max()
	frame = img_as_ubyte(electrons)

	return frame


def add_noise_batch(frames,
					quant_eff=1,
					sigma_dark=10,
					bit_depth=8,
					sensitivity=1,
					baseline=0,
					dyn_range=1000):

	"""Model real camera noise: shot, dark, and discretization

	Parameters
	----------
	"""

	for i in range(len(frames)):
		frames[i] = add_noise(frames[i,0,0,:,:],
							  quant_eff=quant_eff,
							  sigma_dark=sigma_dark,
							  bit_depth=bit_depth,
							  sensitivity=sensitivity,
							  baseline=baseline,
							  dyn_range=dyn_range)

	return frames

def add_psf(frame,
			nphotons,
 			pos_vec,
			ex_wavelen=488,
			em_wavelen=520,
			num_aperture=1.22,
			refr_index=1.333,
			pinhole_rad=.55,
			pinhole_shape='round',
			lat_dim=3):

	"""Adds a point-spread-function (PSF) to a frame.
	   If the pos vector r is 3D, each psf is further blurred
	   in place and then added to the 3D frame.

	Parameters
	----------

	frame : ndarray
		a single frame to be populated
	r : float
		particle coordinate vector
	"""

	args = dict(ex_wavelen=ex_wavelen, \
				em_wavelen=em_wavelen, \
				num_aperture=num_aperture, \
				refr_index=refr_index, \
				pinhole_radius=pinhole_rad, \
				pinhole_shape=pinhole_shape)

	obsvol = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)
	sigma_px, sigma_um = obsvol.sigma.ou

	num_zslices = frame.shape[0]
	x = np.linspace(0, frame.shape[2]-1, frame.shape[2], dtype=np.int)
	y = np.linspace(0, frame.shape[3]-1, frame.shape[3], dtype=np.int)
	x, y = np.meshgrid(x, y)
	h = np.exp(-((x-pos_vec[0])**2/(2*sigma_um**2) + \
				 (y-pos_vec[1])**2/(2*sigma_um**2)))

	h = h*nphotons
	if lat_dim == 3:
		for i in range(num_zslices):
			sigma = .3*abs(i-pos_vec[2])
			#this will add particles in 3d to the 2d lattice as well
			frame[i, :, :, :] += gaussian(h, sigma=sigma)
	else:
		frame += h

	return frame

def add_photon_stats(traj_df, exp_time, photon_rate):

	"""Adds photon statistics column to traj_df

	Parameters
	----------
	"""

	nrecords = traj_df.shape[0]
	lam=exp_time*photon_rate*np.ones(nrecords)
	nphotons = np.random.poisson(lam=exp_time*photon_rate,size=nrecords)
	traj_df['photons'] = nphotons

	return traj_df
