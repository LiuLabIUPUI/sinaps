import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def format_ax(ax,
	xlabel='',
	ylabel='',
	spine_linewidth=2,
	ax_is_box=True,
	xlabel_color=(0,0,0,1),
	ylabel_color=(0,0,0,1),
	label_fontname='Arial',
	label_fontweight='normal',
	label_fontsize='medium',
	xscale=[None,None,None,None],
	yscale=[None,None,None,None],
	tklabel_fontname='Arial',
	tklabel_fontweight='normal',
	tklabel_fontsize='medium',
	show_legend=True,
	legend_loc='upper left',
	legend_frameon=False,
	legend_fontname='Arial',
	legend_fontweight='normal',
	legend_fontsize='medium'):
	"""
	Adjust ax format: axis label, ticker label, tickers.

	Parameters
	----------
	ax : object
		matplotlib ax.

	xlabel : str
		x axis label name.

	ylabel : str
		x axis label name.

	spine_linewidth : int,
		Linewidth of the axis spines

	ax_is_box : bool,
		Determines whether the axis will be a box or just x,y axes

	xlabel_color : tuple
		RGB or RGBA tuple.

	ylabel_color : tuple
		RGB or RGBA tuple.

	xscale : list
		[x_min, x_max, x_major_ticker, x_minor_ticker]

	yscale : list
		[y_min, y_max, y_major_ticker, y_minor_ticker]

	label_fontname : str

	label_fontsize : str or int

	label_fontweight : str or int

	tklabel_fontname : str

	tklabel_fontsize : str or int

	tklabel_fontweight : str or int
	"""

	# """
	# ~~~~~~~~~~~format x, y axis label~~~~~~~~~~~~~~
	# """
	ax.set_xlabel(xlabel,
				color=xlabel_color,
				fontname=label_fontname,
				fontweight=label_fontweight,
				fontsize=label_fontsize)
	ax.set_ylabel(ylabel,
				color=ylabel_color,
				fontname=label_fontname,
				fontweight=label_fontweight,
				fontsize=label_fontsize)

	ax.spines['left'].set_linewidth(spine_linewidth)
	ax.spines['left'].set_color(ylabel_color)

	ax.spines['right'].set_linewidth(spine_linewidth)
	ax.spines['right'].set_color(ylabel_color)

	ax.spines['bottom'].set_linewidth(spine_linewidth)
	ax.spines['bottom'].set_color(xlabel_color)

	ax.spines['top'].set_linewidth(spine_linewidth)
	ax.spines['top'].set_color(xlabel_color)

	if not ax_is_box:
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

	# """
	# ~~~~~~~~~~~format xtick, ytick label~~~~~~~~~~~~~~
	# """
	plt.setp(ax.get_xticklabels(),
				color=xlabel_color,
				fontname=tklabel_fontname,
				fontweight=tklabel_fontweight,
				fontsize=tklabel_fontsize)
	plt.setp(ax.get_yticklabels(),
				color=ylabel_color,
				fontname=tklabel_fontname,
				fontweight=tklabel_fontweight,
				fontsize=tklabel_fontsize)

	ax.tick_params(axis='x', which='both', color=xlabel_color)
	ax.tick_params(axis='y', which='both', color=ylabel_color)

	# """
	# ~~~~~~~~~~~format xlim, ylim, major_tk, minor_tk~~~~~~~~~~~~~~
	# """
	while(len(xscale) < 4):
		xscale.append(None)
	while(len(yscale) < 4):
		yscale.append(None)

	x_min, x_max, x_major_tk, x_minor_tk = xscale
	ax.set_xlim(x_min, x_max)
	if x_major_tk:
		ax.xaxis.set_major_locator(MultipleLocator(x_major_tk))
	if x_minor_tk:
		if x_minor_tk < x_major_tk:
			ax.xaxis.set_minor_locator(MultipleLocator(x_minor_tk))

	y_min, y_max, y_major_tk, y_minor_tk = yscale
	ax.set_ylim(y_min, y_max)
	if y_major_tk:
		ax.yaxis.set_major_locator(MultipleLocator(y_major_tk))
	if y_minor_tk:
		if y_minor_tk < y_major_tk:
			ax.yaxis.set_minor_locator(MultipleLocator(y_minor_tk))

	# """
	# ~~~~~~~~~~~format legend~~~~~~~~~~~~~~
	# """
	if show_legend:
		ax.legend(loc=legend_loc,
				frameon=legend_frameon,
				fontsize=legend_fontsize,
				prop={'family' : legend_fontname,
					'size' : legend_fontsize,
					'weight' : legend_fontweight})

	# """
	# ~~~~~~~~~~~set anchor position~~~~~~~~~~~~~~
	# """
	ax.set_anchor('SW')


def plt2array(fig):
	"""
	Save matplotlib.pyplot figure to numpy rgbndarray.

	Parameters
	----------
	fig : object
		matplotlib figure object.

	Returns
	-------
	rgb_array_rgb: ndarray
		3d ndarray represting the figure

	Examples
	--------
	import matplotlib.pyplot as plt
	import numpy as np
	from cellquantifier.plot.plotutil import plt2array
	t = np.linspace(0, 4*np.pi, 1000)
	fig, ax = plt.subplots()
	ax.plot(t, np.cos(t))
	ax.plot(t, np.sin(t))
	result_array_rgb = plt2array(fig)
	plt.clf()
	plt.close()
	print(result_array_rgb.shape)
	"""

	fig.canvas.draw()
	buf = fig.canvas.tostring_rgb()
	ncols, nrows = fig.canvas.get_width_height()
	rgb_array_rgb = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
	plt.close()

	return rgb_array_rgb

def traj_stats(traj_df, dt=1):

	"""
	Show acceleration, velocity and position statistics

	Parameters
	----------
	"""

	def add_mean(sample, ax):

		mean = '%.2E' % sample.mean()
		var = '%.2E' % sample.var()

		ax.text(.7, .7, mean + ',\n' + var, \
				fontsize=10, transform=ax.transAxes)

	#
	#~~~~~~~Show the acceleration distributions~~~~~~~~~~
	#


	fig, ax = plt.subplots(3, 3, figsize=(10,10))
	ax[0,0].hist(traj_df['a_x'], bins=20, color='purple')
	ax[0,0].set_xlabel(r'$\mathbf{a_{x}}$', fontsize=14)
	ax[0,0].set_yticks([])
	add_mean(traj_df['a_x'], ax[0,0])

	ax[0,1].hist(traj_df['a_y'], bins=20, color='purple')
	ax[0,1].set_xlabel(r'$\mathbf{a_{y}}$', fontsize=14)
	ax[0,1].set_yticks([])
	add_mean(traj_df['a_y'], ax[0,1])

	ax[0,2].hist(traj_df['a_z'], bins=20, color='purple')
	ax[0,2].set_xlabel(r'$\mathbf{a_{z}}$', fontsize=14)
	ax[0,2].set_yticks([])
	add_mean(traj_df['a_z'], ax[0,2])

	#
	#~~~~~~~Show the velocity distributions~~~~~~~~~~
	#

	ax[1,0].hist(traj_df['a_x']*dt, alpha=.5, bins=20, color='red')
	ax[1,0].hist(traj_df['dv_x'], alpha=.5, bins=20, color='blue')
	ax[1,0].set_xlabel(r'$\Delta\mathbf{v_{x}}$', fontsize=14)
	ax[1,0].set_yticks([])
	add_mean(traj_df['dv_x'], ax[1,0])

	ax[1,1].hist(traj_df['a_y']*dt, alpha=.5, bins=20, color='red')
	ax[1,1].hist(traj_df['dv_y'], alpha=.5, bins=20, color='blue')
	ax[1,1].set_xlabel(r'$\Delta\mathbf{v_{y}}$', fontsize=14)
	ax[1,1].set_yticks([])
	add_mean(traj_df['dv_y'], ax[1,1])

	ax[1,2].hist(traj_df['a_z']*dt, alpha=.5, bins=20, color='red')
	ax[1,2].hist(traj_df['dv_z'], alpha=.5, bins=20, color='blue')
	ax[1,2].set_xlabel(r'$\Delta\mathbf{v_{z}}$', fontsize=14)
	ax[1,2].set_yticks([])
	add_mean(traj_df['dv_z'], ax[1,2])

	# """
	# ~~~~~~~Show the displacement distributions~~~~~~~~~~
	# """

	ax[2,0].hist(traj_df['dv_x']*dt, alpha=.5, bins=20, color='red')
	ax[2,0].hist(traj_df['dx'], alpha=.5, bins=20, color='blue')
	ax[2,0].set_xlabel(r'$\Delta\mathbf{x}$', fontsize=14)
	ax[2,0].set_yticks([])
	add_mean(traj_df['dx'], ax[2,0])

	ax[2,1].hist(traj_df['dv_y']*dt, alpha=.5, bins=20, color='red')
	ax[2,1].hist(traj_df['dy'], alpha=.5, bins=20, color='blue')
	ax[2,1].set_xlabel(r'$\Delta\mathbf{y}$', fontsize=14)
	ax[2,1].set_yticks([])
	add_mean(traj_df['dy'], ax[2,1])

	ax[2,2].hist(traj_df['dv_z']*dt, alpha=.5, bins=20, color='red')
	ax[2,2].hist(traj_df['dz'], alpha=.5, bins=20, color='blue')
	ax[2,2].set_xlabel(r'$\Delta\mathbf{z}$', fontsize=14)
	ax[2,2].set_yticks([])
	add_mean(traj_df['dz'], ax[2,2])

def show_traj(ax, traj_df, ndim, color='red'):

	"""Plots a single trajectory in 2D or 3D

	Parameters
	----------

	ax: axis object
		Matplotlib axis object to use for plotting

	traj_df : DataFrame
		Contains 'x', 'y', 'particle' columns
	"""

	if ndim == 2:
			ax.plot(traj_df['x'], \
					traj_df['y'], \
					linewidth=1,
					color=color)
	if ndim == 3:
			ax.plot(traj_df['x'], \
					traj_df['y'], \
					traj_df['z'], \
					linewidth=1,
					color=color)

def show_traj_batch(traj_df, origins=None, ndim=3):

	"""Plots the trajectories in 2D or 3D

	Parameters
	----------
	origins : 1D ndarray
		Contains the origin of each particle e.g. lattice, ring, etc.
		that can be used to show the origins of the particles

	"""

	nparticles = traj_df['particle'].nunique()

	fig = plt.figure()
	if ndim == 3:
		ax = fig.gca(projection='3d')
	else:
		ax = fig.gca()

	colors = plt.cm.get_cmap('viridis')
	colors = colors(np.linspace(0, 1, nparticles))
	for i in range(nparticles):
		this_df = traj_df.loc[traj_df['particle'] == i]
		show_traj(ax, this_df, ndim, color=colors[i])

	tst = traj_df.loc[traj_df['frame'] == 0]
	if origins is not None:
		ax.scatter(tst['x'],tst['y'], c='r', s=20)

	format_ax(ax,
			  xlabel=r'$\mathbf{x}$',
			  ylabel=r'$\mathbf{y}$',
			  show_legend=False,
			  label_fontsize=15)
	plt.show()

def show_force_comp(traj_df):

	"""Show x,y,z components of the stochastic force

	Parameters
	----------

	"""

	# """
	# ~~~~~~~Initialize the figure~~~~~~~~~~
	# """

	nparticles = traj_df['particle'].nunique()
	fig,ax = plt.subplots(1,3, figsize=(15,5))
	colors = plt.cm.get_cmap('coolwarm')
	colors = colors(np.linspace(0, 1, nparticles))


	# """
	# ~~~~~~~Plot the components~~~~~~~~~
	# """

	for n in range(nparticles):
		this_df = traj_df.loc[traj_df['particle'] == n]

		a_x = this_df['a_x'].to_numpy()
		a_y = this_df['a_y'].to_numpy()
		a_z = this_df['a_z'].to_numpy()

		ax[0].plot(a_x*1e6, color=colors[n], alpha=1)
		ax[1].plot(a_y*1e6, color=colors[n], alpha=1)
		ax[2].plot(a_z*1e6, color=colors[n], alpha=1)

	labels = [r'$\mathbf{\xi_{x}/m(\frac{\mu m}{s^{2}})}$',\
			  r'$\mathbf{\xi_{y}/m(\frac{\mu m}{s^{2}})}$',\
			  r'$\mathbf{\xi_{z}/m(\frac{\mu m}{s^{2}})}$']

	for i, x in enumerate(ax):
		x = format_ax(x,
					  xlabel=r'$\mathbf{Time}$',
					  ylabel=labels[i],
					  label_fontsize=15,
					  show_legend=False,
					  ax_is_box=False)
	plt.tight_layout()
	plt.show()

def save_animation(frames):

	"""Save a matplotlib ArtistAnimation object of the simulation

	Parameters
	----------
	"""

	fig, ax = plt.subplots()
	nframes = len(frames)
	ims = []
	for i in range(nframes):

		im = plt.imshow(self.frames[i], animated=True, cmap='coolwarm')
		ims.append([im])

	ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
									repeat_delay=1000)


	from matplotlib.animation import FFMpegWriter
	writer = FFMpegWriter(fps=15,
						  codec='libx264',
						  metadata=dict(artist='Me'),
						  bitrate=1800)
	ani.save("movie.mp4", writer=writer)
	plt.show()
