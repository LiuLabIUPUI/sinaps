import psf
import numpy as np
from skimage.filters import gaussian

def add_psfs(frame,
			 traj_df,
			 ex_wavelen=488,
			 em_wavelen=520,
			 num_aperture=1.22,
			 refr_index=1.333,
			 pinhole_rad=.55,
			 pinhole_shape='round',
			 lat_dim=3):

	"""Adds point-spread-functions (PSF) to a single frame.

	Parameters
	----------

	frame : ndarray
		a single frame to be populated
	r : float
		particle coordinate vector
	"""

	args = dict(ex_wavelen=ex_wavelen, em_wavelen=em_wavelen,
				num_aperture=num_aperture, refr_index=refr_index,
				pinhole_radius=pinhole_rad, pinhole_shape=pinhole_shape)

	obsvol = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)
	sigma_px, sigma_um = obsvol.sigma.ou
	particles = traj_df['particle'].unique()

	xsize, ysize = frame.shape[2:4]
	x = np.linspace(0, xsize-1, xsize, dtype=np.int)
	y = np.linspace(0, ysize-1, ysize, dtype=np.int)
	x, y = np.meshgrid(x, y)

	for particle in particles:

		pos = traj_df.loc[traj_df['particle'] == particle, ['x','y']].to_numpy()
		pos += np.round(frame.shape[3]/2); x0,y0 = pos[0]
		_psf = np.exp(-((x-x0)**2/(2*sigma_um**2) + (y-y0)**2/(2*sigma_um**2)))

		nphotons = traj_df.loc[traj_df['particle'] == particle, \
									   'photons'].to_numpy()
		_psf = _psf*nphotons[0]; frame += _psf

	return frame

def add_psfs_batch(frames,
				   traj_df,
				   ex_wavelen=488,
				   em_wavelen=520,
				   num_aperture=1.22,
				   refr_index=1.333,
				   pinhole_rad=.55,
				   pinhole_shape='round',
				   lat_dim=3):

	"""Adds point-spread-functions (PSF) to a movie.
	   This is a batch function that calls add_psfs() for
	   each frame of the movie

	Parameters
	----------

	frames : ndarray
		a stack of frames representing a movie
	traj_df : DataFrame
		DataFrame with particle, frame, x, y , nphotons columns
	"""

	nframes = len(frames)
	for n in range(nframes):
		this_df = traj_df.loc[traj_df['frame'] == n]
		frames[n] = add_psfs(frames[n],
					 this_df,
					 ex_wavelen=ex_wavelen,
					 em_wavelen=em_wavelen,
					 num_aperture=num_aperture,
					 refr_index=refr_index,
					 pinhole_rad=pinhole_rad,
					 pinhole_shape=pinhole_shape,
					 lat_dim=lat_dim)

	return frames

def add_noise(frame, sigma_dark=10, bit_depth=8, baseline=0):

	"""Model real camera noise

	Parameters
	----------
	"""

	# Add dark noise
	frame = np.random.normal(scale=sigma_dark, \
							  size=frame.shape) + frame
	# Convert to ADU and add baseline
	max_adu     = np.int(2**bit_depth - 1)
	frame = frame.astype(np.int)
	frame += baseline
	frame[frame > max_adu] = max_adu #pixel saturation

	return frame


def add_noise_batch(frames,
					sigma_dark=10,
					bit_depth=8,
					baseline=0):

	nframes = len(frames)
	for n in range(nframes):
		frames[n] = add_noise(frames[n],
							  sigma_dark=sigma_dark,
							  bit_depth=bit_depth,
							  baseline=baseline)

	return frames

def add_photon_stats(traj_df, exp_time=1, photon_rate=1):

	"""Adds photon statistics column to traj_df

	Parameters
	----------
	"""

	nrecords = traj_df.shape[0]
	lam=exp_time*photon_rate*np.ones(nrecords)
	nphotons = np.random.poisson(lam=exp_time*photon_rate,size=nrecords)
	traj_df['photons'] = nphotons

	return traj_df
