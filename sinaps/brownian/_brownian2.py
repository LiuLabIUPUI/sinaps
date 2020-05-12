from math import sqrt
from scipy.stats import norm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def build_traj2(nparticles,
				nframes,
				ndim=2,
				dcoeff=10,
				origins=None):

	"""Build a set of trajectories based on gaussian distributed displacements

	Parameters
	----------
	framerate: float,
			   frames per second
	"""

	columns = ['frame', 'particle','x', 'y','dx','dy']
	traj_df = pd.DataFrame(columns=columns)

	for n in range(nframes):

		scale = np.sqrt(4*dcoeff*n)
		this_df = pd.DataFrame(columns=columns)
		this_df['particle'] = np.arange(0,nparticles,1)
		this_df = this_df.assign(frame=n)

		this_df['dx'] = norm.rvs(size=nparticles, scale=scale)
		this_df['dy'] = norm.rvs(size=nparticles, scale=scale)

		traj_df = traj_df.append(this_df)

	for n in range(nparticles):

		this_df = traj_df.loc[traj_df['particle'] == n]
		x0,y0 = (0,0)

		if origins is not None:
			x0,y0 = origins[n]
		this_df['x'] = this_df['dx'].cumsum() + x0
		this_df['y'] = this_df['dy'].cumsum() + y0
		traj_df.loc[traj_df['particle'] == n] = this_df

	return traj_df
