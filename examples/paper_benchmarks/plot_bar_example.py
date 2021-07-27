import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

import numpy as np
import pandas as pd

sample_data = []
num_bars=4
sample_data = np.random.rand(num_bars)


plt.figure(figsize=(8, 6))#, dpi=150)

cmap = matplotlib.cm.get_cmap('tab20')

plt.bar(1, sample_data[0],
        label="Data 1" color=cmap.colors[1],
        edgecolor=cmap.colors[0], linewidth=2
       )
plt.bar(2, sample_data[1],
        label="Data 2", color=cmap.colors[3],
        edgecolor=cmap.colors[2], linewidth=2, hatch="x")

plt.bar(3, sample_data[2],
        label="Data 3", color=cmap.colors[5],
        edgecolor=cmap.colors[4], linewidth=2, hatch="//")

plt.bar(4, sample_data[3],
        label="Data 4", color=cmap.colors[7],
        edgecolor=cmap.colors[6], linewidth=2, hatch="\\\\")

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.yticks(fontsize=28)
plt.ylabel(r"Contraction time (s)", fontsize=20)

plt.legend(fontsize=14, loc='upper left', #bbox_to_anchor=(0.5, -0.45),
          ncol=1, fancybox=True, shadow=False)

ax = plt.gca()
rects = ax.patches

# Make some labels.
labels = [f"{rects[i].get_height():.2f}" for i in range(len(rects))]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 0.25, label,
            ha='center', va='bottom', fontsize=20)

plt.ylim([0, 10])
plt.tight_layout()
plt.savefig("runtimes.pdf")
