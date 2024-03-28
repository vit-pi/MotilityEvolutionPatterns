###
# IMPORTS
###
###
# IMPORTS
###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Pattern as pt

###
# PARAMETERS
###
# External: By varying the truth value, Fig. 2 (true) and Fig. S3 (false) are reproduced
eco_forcing = False

# Internal
distant_mutations = False
if eco_forcing:
    if distant_mutations:
        filename = "PreyPred_LongMut"
        t_max = 500
        plot_times = [100, 200, 300, 400]
    else:
        filename = "PreyPred_ShortMut"
        t_max = 1400
        plot_times = [100, 300, 870, 1200]
else:
    if distant_mutations:
        filename = "CoopDef_LongMut"
        t_max = 500
        plot_times = [100, 200, 300, 400]
    else:
        filename = "CoopDef_ShortMut"
        t_max = 1400
        plot_times = [100, 400, 700, 1100]
seed = 0
d0 = 0.1
d1 = 1
colors = ["#214478ff", "#aa0000ff"]    # [blue, red]

###
# CREATE AND FORMAT FIGURE
###
fig = plt.figure(figsize=(10, 6.7))
gs = gridspec.GridSpec(3, 5, figure=fig, wspace=0.4, hspace=0.4, width_ratios=(0.3, 3, 3, 3, 3))
ax_fitness = fig.add_subplot(gs[2,:])
if eco_forcing:
    ax_fitness.set_ylim(-50, 20)
    ax_fitness.set_yticks([-50, -40, -30, -20, -10, 0, 10, 20])
    ax_fitness.set_yticklabels([None, -40, None, -20, None, 0, None, 20])
else:
    ax_fitness.set_ylim(-15, 5)
    ax_fitness.set_yticks([-15, -10, -5, 0, 5])
ax_fitness.set_xticks([0]+plot_times[0:4])
ax_fitness.set_ylabel("total fitness $F_i$", fontsize=12)
ax_fitness.set_xlabel("time $t$", fontsize=12)
for time in plot_times:
    ax_fitness.axvline(x=time, color='black')
ax_blue = []
ax_red = []
ax_colorbars = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
for snap in range(4):
    # add panels
    if snap == 0:
        ax_blue.append(fig.add_subplot(gs[0, snap+1]))
        ax_red.append(fig.add_subplot(gs[1, snap+1], sharex=ax_blue[snap]))
    else:
        ax_blue.append(fig.add_subplot(gs[0, snap + 1], sharey=ax_blue[0]))
        ax_red.append(fig.add_subplot(gs[1, snap + 1], sharex=ax_blue[snap], sharey=ax_blue[0]))
    # add labels
    ax_red[snap].set_xlabel("space $x$", fontsize=12)
ax_red[0].set_ylabel("motility $d_I$", fontsize=12, color=colors[1])
ax_blue[0].set_ylabel("motility $d_A$", fontsize=12, color=colors[0])
###
# RUN THE SIMULATION AND MAKE A SNAPSHOT
###
# Create a pattern object
env_prop = pt.EnvProp()
if eco_forcing:
    env_prop.int_fitness = pt.IntFit1(2,0.62,0.5)
    env_prop.pos_num = 201
else:
    env_prop.int_fitness = pt.IntFit2(2.4,8,1,1.2)
    env_prop.pos_num = 201
env_prop.distant_mutations = distant_mutations
env_prop.fitness_memory_time = t_max
env_prop.initialize()
spec_prop = [pt.SpecProp(), pt.SpecProp()]
for spec in range(2):
    spec_prop[spec].diff_num = 11
    spec_prop[spec].mut_rate = 1/150
    spec_prop[spec].initialize()
pattern = pt.Pattern(env_prop, spec_prop, pt.init_pattern(env_prop, spec_prop, 1, seed, 1, d0, d1))
# Update pattern object up until t_max & make plot
snap = 0
while pattern.time < t_max:
    pattern.fitness_update()
    pattern.mutation_update()
    if snap < len(plot_times) and pattern.time >= plot_times[snap]:
        print("Snapshot: " + str(snap))
        # Plot pattern heatmaps
        im = pt.plot_heatmap_snapshot(pattern, [ax_blue[snap], ax_red[snap]], False)
        # Update snap
        snap += 1
    pattern.update_pattern_euler()
pattern.time = np.floor(pattern.time)
# Plot fitness
pt.plot_fitness(pattern, 0, ax_fitness, True, False)
# Plot color bars
if eco_forcing:
    labels = ["prey $n_A$", "predators $n_I$"]
else:
    labels = ["cooperators $n_A$", "defectors $n_I$"]
for spec in range(2):
    vmin, vmax = im[spec].get_clim()
    plt.colorbar(im[spec], cax=ax_colorbars[spec], ticks=[0, vmax])
    ax_colorbars[spec].set_ylabel(labels[spec], fontsize=12, labelpad=-60, color=colors[spec])
    ax_colorbars[spec].yaxis.set_ticks_position('left')
    ax_colorbars[spec].tick_params(axis='both', which='major', labelsize=12)
# Plot
fig.savefig("Figures/Sketches/Fig2_"+filename+".svg")
plt.show()
