import numpy as np
import pandas as pd
import warnings
from cellquantifier.smt import *

def det_fit_track(pims_frames,
				min_sig=1,
				max_sig=3,
				num_sig=5,
				blob_thres=0.1,
				peak_thres_rel=0.1,
				r_to_sigraw=3,
				diagnostic=False,
				pltshow=False,
				search_range=3,
				memory=5,
				pixel_size=1,
				frame_rate=1,
				divide_num=1,
				filters=None,
				do_filter=False,
				traj_len_thres=None):

	"""Wrapper for cellquantifier detection, psf fitting, and tracking functions

	Parameters
	----------
	frames: ndarray,
			A movie of stationary particles
	"""

	blobs_df, pltarray = detect_blobs_batch(pims_frames,
									min_sig=min_sig,
									max_sig=max_sig,
									num_sig=num_sig,
									blob_thres=blob_thres,
									peak_thres_rel=peak_thres_rel,
									r_to_sigraw=r_to_sigraw,
									pixel_size=pixel_size,
									diagnostic=diagnostic,
									pltshow=pltshow)

	blobs_df, pltarray = fit_psf_batch(pims_frames,
								   blobs_df,
								   diagnostic=diagnostic,
								   pltshow=pltshow)

	if not traj_len_thres:
		traj_len_thres = int(round(len(pims_frames)/2))

	blobs_df, im = track_blobs(blobs_df,
								   search_range=search_range,
								   memory=memory,
								   pixel_size=pixel_size,
								   frame_rate=frame_rate,
								   divide_num=divide_num)

	blobs_df = tp.filter_stubs(blobs_df, traj_len_thres)

	return blobs_df

def get_msd(df, pixel_size=1, frame_rate=1):

	all_std = pd.DataFrame()
	all_msd = pd.DataFrame()
	particles = df['particle'].unique()
	for particle in particles:
		this_df = df.loc[df['particle'] == particle]
		msd = msd_gaps(this_df, mpp=pixel_size, fps=frame_rate)
		all_std[particle] = msd['std_msd']
		all_msd[particle] = msd['msd'] #msd

	all_std = np.mean(all_std, axis=1)

	return all_msd, all_std

def fit_brownian(t, msd, divide_num=1):

	m = int(round(len(msd)/divide_num))
	t = msd.index[:m]
	msd = msd[:m]
	tmp = t[:,np.newaxis]
	a, _, _, _ = np.linalg.lstsq(tmp, msd)
	dcoeff = a[0]/2
	return dcoeff

def get_loc_bias(blobs_df):


	particles = blobs_df['particle'].unique()
	nparticles = len(particles)
	loc_bias = np.zeros((nparticles,2))

	for i,particle in enumerate(particles):
		this_df = blobs_df.loc[blobs_df['particle'] == particle]
		xdiff = np.abs(np.mean(np.diff(this_df['x'])))
		ydiff = np.abs(np.mean(np.diff(this_df['y'])))
		loc_bias[i] = [xdiff,ydiff]

	loc_bias = np.sqrt(np.sum(loc_bias**2, axis=1))

	return loc_bias

def msd_iter(pos, lagtimes):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		for lt in lagtimes:
			diff = pos[lt:] - pos[:-lt]
			yield np.concatenate((np.nanmean(diff, axis=0),
								  np.nanmean(diff**2, axis=0),
								  np.nanvar(diff**2, axis=0)))

def msd_gaps(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=None):
	"""Compute the mean displacement and mean squared displacement of one
	trajectory over a range of time intervals."""
	if pos_columns is None:
		pos_columns = ['x', 'y']
	result_columns = ['<{}>'.format(p) for p in pos_columns] + \
					 ['<{}^2>'.format(p) for p in pos_columns] + \
					 ['var_{}^2'.format(p) for p in pos_columns]


	pos = traj.set_index('frame')[pos_columns] * mpp
	max_lagtime = min(max_lagtime, len(pos) - 1)  # checking to be safe
	lagtimes = np.arange(1, max_lagtime + 1)

	result = pd.DataFrame(msd_iter(pos.values, lagtimes),
						  columns=result_columns, index=lagtimes)

	plt.plot(result['var_x^2'])
	plt.show()
	result['msd'] = result[result_columns[len(pos_columns):2*len(pos_columns)]].sum(1)
	result['std_msd'] = np.sqrt(result['var_x^2'] + result['var_y^2'])
	result['md'] = np.sqrt(result['msd'])

	if detail:
		# effective number of measurements
		# approximately corrected with number of gaps
		result['N'] = _msd_N(len(pos), lagtimes) * len(traj) / len(pos)
	result['lagt'] = result.index.values/float(fps)
	result.index.name = 'lagt'
	return result
