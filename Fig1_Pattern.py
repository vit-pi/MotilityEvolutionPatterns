###
# IMPORTS
###
import matplotlib.pyplot as plt
import Pattern as pt

###
# PARAMETERS
###
# External: patterned (true,false) to reproduce the corresponding steady state in Fig 1d
patterned = True
# Internal
t_max = 1000
if patterned:
    d0 = 0.1
else:
    d0 = 1
d1 = 1
colors = ["#214478ff", "#aa0000ff"]    # [blue, red]

###
# RUN THE SIMULATION AND MAKE A SNAPSHOT
###
# Create a pattern object
env_prop = pt.EnvProp()
spec_prop = [pt.SpecProp(), pt.SpecProp()]
spec_prop[0].diff_max = d0
spec_prop[0].diff_min = d0
spec_prop[0].initialize()
spec_prop[1].diff_max = d1
spec_prop[1].diff_min = d1
spec_prop[1].initialize()
pattern = pt.Pattern(env_prop, spec_prop, pt.init_pattern(env_prop, spec_prop, 0, None, 2, d0, d1, t_max))
# Create figure
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(1.5, 1), constrained_layout=True)
# Update pattern object up until t_max & make plot
pt.plot_heatmap_snapshot(pattern, axs, False)
# Modify labels
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[0].set_ylabel("$n_A$", rotation=0, ha='right', va="center", fontsize=12, color=colors[0])
axs[1].set_ylabel("$n_I$", rotation=0, ha='right', va="center", fontsize=12, color=colors[1])
axs[1].set_xlabel("space $x$", fontsize=12)

# Show plot
if patterned:
    fig.savefig("Figures/Sketches/Fig1_PatternP.svg")
else:
    fig.savefig("Figures/Sketches/Fig1_PatternH.svg")
plt.show()
