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

left_hand_indices = 44
right_hand_indices = 53
left_feet_indices = 9
right_feet_indices = 25

ara = np.arange(2, length + 2, 1)

# plot hand curve
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

x = root_state[:, left_hand_indices, 0]
y = root_state[:, left_hand_indices, 1]
z = root_state[:, left_hand_indices, 2]
norm = mcolors.Normalize(vmin=-10, vmax=90, clip=True)
scatter1 = ax.scatter(x, y, z, c=ara, label='base', cmap='OrRd', marker='o', norm=norm)

x = root_state[:, right_hand_indices, 0]
y = root_state[:, right_hand_indices, 1]
z = root_state[:, right_hand_indices, 2]
scatter2 = ax.scatter(x, y, z, c=ara, label='base', cmap='OrRd', marker='o', norm=norm)

# plot feet curve
norm = mcolors.Normalize(vmin=-10, vmax=90, clip=True)
x = root_state[:, left_feet_indices, 0]
y = root_state[:, left_feet_indices, 1]
z = root_state[:, left_feet_indices, 2]
scatter3 = ax.scatter(x, y, z, c=ara, label='base', cmap='Blues', marker='o', norm=norm)

x = root_state[:, right_feet_indices, 0]
y = root_state[:, right_feet_indices, 1]
z = root_state[:, right_feet_indices, 2]
scatter4 = ax.scatter(x, y, z, c=ara, label='base', cmap='Blues', marker='o', norm=norm)

#ax.legend()
ax.w_xaxis.pane.fill = True
ax.w_xaxis.pane.set_facecolor((1, 1, 1))

ax.w_yaxis.pane.fill = True
ax.w_yaxis.pane.set_facecolor((1, 1, 1))

ax.w_zaxis.pane.fill = True
ax.w_zaxis.pane.set_facecolor((1, 1, 1))

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.3))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.3))


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
ax.legend(legends,['Hands', 'Feet'], loc='lower right', fontsize=12,  bbox_to_anchor=(0.9, 0.6))

# legend
blue_star = mlines.Line2D([], [], color='blue', marker='*', linestyle='None',
                          markersize=10, label='Blue stars')
red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='Red squares')

fig.tight_layout()
ax.view_init(elev=10, azim=-100)

fig.savefig(f'./visualization/output/3d_trajectory_hand_feet_g1_{args.terrain}.pdf', dpi=300, bbox_inches='tight')
plt.show()