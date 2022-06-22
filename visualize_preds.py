import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from tqdm import tqdm
plt.ion()

# pred_path = './vo_results/09.txt'
pred_path = '/home/tuannghust/SC-SfMLearner-Release/out_square/new_square_rgb.txt'

out_odom = np.loadtxt(pred_path)

fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
plt.show()
verts = [(0,0)]
codes = [Path.MOVETO]
path = Path(verts, codes)
patch = patches.PathPatch(path, facecolor='white', lw=2)
ax.add_patch(patch)

for i in tqdm(range(len(out_odom))):
    odom = out_odom[i]
    odom = np.reshape(odom, newshape=[3,4])
    x = odom[0, 3]
    z = odom[2, 3]

    verts = verts + [(x, z)]
    codes = codes + [Path.LINETO]

    path  = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='white', lw=2)
    ax.add_patch(patch)
    plt.draw()
    plt.show(block=False)
    plt.pause(0.05)

