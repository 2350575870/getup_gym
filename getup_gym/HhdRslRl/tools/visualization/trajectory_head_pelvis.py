import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--terrain', type=str, default='ground')
args = parser.parse_args()
os.makedirs('./visualization/output', exist_ok=True)

sns.set_theme(style='whitegrid')
sns.set_style({'axes.facecolor': 'FFFFFF', "grid": False})

length = 75
root_state = np.load(f"/home/bitbot/getup_gym_project/visualization/data/Getup_root_state.npy", allow_pickle=True)[:length]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

head_indices = 35
pelvis_indices = 0
ara = np.arange(2, length + 2, 1)

# plot head curve
fig = plt.figure(figsize=(8,6))
# plt.tick_params(left = False, bottom = False) 
ax = fig.add_subplot(projection='3d')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

x = root_state[:, head_indices, 0]
y = root_state[:, head_indices, 1]
z = root_state[:, head_indices, 2]

norm = mcolors.Normalize(vmin=-10, vmax=90, clip=True)

scatter1 = ax.scatter(x, y, z, c=ara, label='Head', cmap='OrRd', marker='o', norm=norm)
#ax.legend()
ax.w_xaxis.pane.fill = True
ax.w_xaxis.pane.set_facecolor((1, 1, 1))

ax.w_yaxis.pane.fill = True
ax.w_yaxis.pane.set_facecolor((1, 1, 1))

ax.w_zaxis.pane.fill = True
ax.w_zaxis.pane.set_facecolor((1, 1, 1))

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.3))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.zaxis.set_major_locator(ticker.MultipleLocator(0.3))
# plot pevlis curve
norm = mcolors.Normalize(vmin=-10, vmax=90, clip=True)

x = root_state[:, pelvis_indices, 0]
y = root_state[:, pelvis_indices, 1]
z = root_state[:, pelvis_indices, 2]

scatter2 = ax.scatter(x, y, z, c=ara, label='Base', cmap='Blues', marker='o', norm=norm)

from matplotlib.lines import Line2D
color_red   = '#A94A4A'
color_green = '#889E73'
#color_blue  = (0.297, 0.5, 0.9)
color_blue = '#418BC0'
color_yellow = '#F4D793'
color_black = '#9D6F66'
color_grey = '#929292'
color_white = '#982176'
color_brown = "#AF8260"
legends = []
enmax_palette = [color_blue, color_red, color_grey, color_brown, color_green, color_black, color_grey]
legends.append(Line2D([0], [0], color=color_red, linewidth=3, linestyle='-'))
legends.append(Line2D([0], [0], color=color_blue, linewidth=3, linestyle='-'))
ax.legend(legends,['Head', 'Base'], loc='lower right', fontsize=12,  bbox_to_anchor=(0.45, 0.6))

# legend
blue_star = mlines.Line2D([], [], color='blue', marker='*', linestyle='None',
                          markersize=10, label='Blue stars')
red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='Red squares')

ax.view_init(elev=5, azim=-110)

fig.tight_layout()
fig.savefig(f'./visualization/output/3d_trajectory_head_pelvis_g1_{args.terrain}.pdf', dpi=300, bbox_inches='tight')
plt.show()
