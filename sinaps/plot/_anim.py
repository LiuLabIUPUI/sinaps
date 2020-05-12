from matplotlib import animation
from copy import deepcopy
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np

# Initialize the figure with an empty frame
def init():
    img.set_data(np.zeros((nframes, nframes)))
    return img,

# This function is called for each frame of the animation
def animate(frame_index):
    frame = _frames[frame_index]
    img.set_data(frame)
    return img,

# Create the animation
def animate_frames(frames):

    global img
    global nframes
    global _frames

    _frames = deepcopy(frames)
    nframes = len(frames)
    fig,ax = plt.subplots()
    img = ax.imshow(_frames[0], cmap='coolwarm', vmin=0)
    anim = animation.FuncAnimation(fig, animate, frames=nframes,
                                   init_func=init, interval=20, blit=True)

    return anim
