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
# No External parameters - Fig 4 is reproduced
# Internal
t_max = 3000
seed = 0
d0 = 0.1
d1 = 1
fitness_memory_time = 50
eco_forcing = [True, False, True]
distant_mutations = [False, False, True]

###
# CREATE AND FORMAT FIGURE
###
fig = plt.figure(figsize=(11, 7.5))
gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.5, hspace=0.5)
ax_blue = []
ax_red = []
ax_fitness = []
ax_diffplane = []
ax_diffblue = []
ax_diffred = []
for plot in range(3):
    # add panels
    ax_blue.append(fig.add_subplot(gs[plot, 0]))
    ax_red.append(fig.add_subplot(gs[plot, 1]))
    ax_fitness.append(fig.add_subplot(gs[plot, 2]))
    sub_gs = gs[plot, 3].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    ax_diffplane.append(fig.add_subplot(sub_gs[1, 0]))
    ax_diffblue.append(fig.add_subplot(sub_gs[0, 0], sharex=ax_diffplane[plot]))
    ax_diffred.append(fig.add_subplot(sub_gs[1, 1], sharey=ax_diffplane[plot]))
    # add labels
    ax_diffplane[plot].set_xlabel("diffusivity $D_A$", fontsize=12)
    ax_diffplane[plot].set_ylabel("diffusivity $D_I$", fontsize=12)
    ax_red[plot].set_xlabel("space $x$", fontsize=12)
    ax_red[plot].set_ylabel("diffusivity $D_I$", fontsize=12)
    ax_blue[plot].set_xlabel("space $x$", fontsize=12)
    ax_blue[plot].set_ylabel("diffusivity $D_A$", fontsize=12)
    ax_fitness[plot].set_ylabel("total fitness $F_i$", fontsize=12)
    ax_fitness[plot].set_xlabel("time $t$", fontsize=12)
    ax_fitness[plot].set_xticks([t_max-fitness_memory_time, t_max])

###
# RUN THE SIMULATION AND MAKE A SNAPSHOT
###
# Create a pattern object
env_prop = [pt.EnvProp() for i in range(3)]
spec_prop = [[pt.SpecProp(), pt.SpecProp()] for i in range(3)]
for plot in range(3):
    if eco_forcing[plot]:
        env_prop[plot].int_fitness = pt.IntFit1(2, 0.62, 0.5)
    else:
        env_prop[plot].int_fitness = pt.IntFit2(2.4, 8, 1, 1.2)
    env_prop[plot].distant_mutations = distant_mutations[plot]
    env_prop[plot].fitness_memory_time = fitness_memory_time
    for spec in range(2):
        spec_prop[plot][spec].diff_num = 11
        spec_prop[plot][spec].mut_rate = 5e-3
        spec_prop[plot][spec].initialize()
# Update pattern object up until t_max & make plot
for plot in range(3):
    # Initialize the pattern
    np.random.seed(seed)
    pattern = pt.Pattern(env_prop[plot], spec_prop[plot], pt.init_pattern(env_prop[plot], spec_prop[plot], 1, None, 1, d0, d1))
    # Update the pattern
    while pattern.time < t_max-env_prop[plot].time_step:
        pattern.fitness_update()
        pattern.mutation_update()
        pattern.update_pattern_euler()
    # Plot the heatmaps
    pt.plot_heatmap_snapshot(pattern, [ax_blue[plot], ax_red[plot]], False)
    # Plot distribution of diffusivities
    pt.plot_diffusivity_distribution(pattern, [ax_diffplane[plot], ax_diffblue[plot], ax_diffred[plot]], False)
    # Plot fitness
    pt.plot_fitness(pattern, 0, ax_fitness[plot], True, False)
    if plot != 1:
        ax_fitness[plot].set_ylim([-5, 5])
        ax_fitness[plot].set_yticks([-5, 0, 5])
    else:
        ax_fitness[plot].set_ylim([-15, 15])
        ax_fitness[plot].set_yticks([-15, 0, 15])
# Plot show and save
fig.savefig("Figures/Sketches/Fig4_ESS.svg")
plt.show()
