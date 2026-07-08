import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
})

categories = ['Race/ethnicity\n(n = 9)', 'Random\n(n = 9)', 'Random\n(n = 50)']
powers = [1.0, 0.048, 0.292]
colors = ['#2171b5', '#cb181d', '#cb181d']

fig, ax = plt.subplots(figsize=(5, 3.5))
fig.subplots_adjust(top=0.82)
ax.set_title('Grouping strategy determines ecological power', pad=20)

bars = ax.barh(categories[::-1], powers[::-1], height=0.5,
               color=colors[::-1], edgecolor='white', linewidth=0.5)

ax.axvline(x=0.80, color='#666666', linestyle='--', linewidth=0.8, zorder=0)
ax.axvline(x=0.05, color='#999999', linestyle=':', linewidth=0.8, zorder=0)

ax.text(0.80, 1.03, '80%', fontsize=8, color='#666666', ha='center',
        transform=ax.get_xaxis_transform(), clip_on=False)
ax.text(0.05, 1.03, '5%', fontsize=8, color='#999999', ha='center',
        transform=ax.get_xaxis_transform(), clip_on=False)

for bar, val in zip(bars, powers[::-1]):
    x_pos = val + 0.02
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{val:.0%}' if val == 1.0 else f'{val:.1%}',
            ha='left', va='center', fontsize=10, fontweight='bold',
            color=bar.get_facecolor())

ax.set_xlim(0, 1.15)
ax.set_xlabel('Statistical power (2,000 simulations)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.15, linewidth=0.5)
ax.tick_params(axis='y', length=0)

plt.tight_layout()
plt.savefig('fig2_power.pdf', bbox_inches='tight')
plt.savefig('fig2_power.png', bbox_inches='tight', dpi=300)
print("Saved fig2_power.pdf and .png")
