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
# External: To reproduce each figure, var the type: Fig. 4 (type=0), Fig. S2 (type=1), Fig. S5 (type=2)
type = 0

# Internal
seed = 0
fitness_memory_time = 50
colors = ["#214478ff", "#aa0000ff"]    # [blue, red]
if type == 0:
    plot_num = 3
    eco_forcing = True
    d_val = [[0.1,1],[0.5,0.1],[0.5,1]]
    distant_mutations = False
    t_max = [3000, 3000, 3000]
elif type == 1:
    plot_num = 2
    eco_forcing = True
    d_val = [[0.1, 1], [0.5, 0.1], [0.5, 1]]
    distant_mutations = True
    t_max = [3000, 3000, 3000]
else:
    plot_num = 3
    eco_forcing = False
    d_val = [[0.1, 1], [0.5, 0.1], [0.5, 1]]
    distant_mutations = False
    t_max = [3000, 3000,3000]


###
# CREATE AND FORMAT FIGURE
###
fig = plt.figure(figsize=(11, 2.5*plot_num))#, constrained_layout=True)
gs = gridspec.GridSpec(plot_num, 4, figure=fig, wspace=0.5, hspace=0.5)
ax_blue = []
ax_red = []
ax_fitness = []
ax_diffplane = []
ax_diffblue = []
ax_diffred = []
ax_abundance = []
for plot in range(plot_num):
    # add panels
    ax_blue.append(fig.add_subplot(gs[plot, 1]))
    ax_red.append(fig.add_subplot(gs[plot, 2]))
    sub_gs0 = gs[plot, 0].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    ax_diffplane.append(fig.add_subplot(sub_gs0[1, 0]))
    ax_diffblue.append(fig.add_subplot(sub_gs0[0, 0], sharex=ax_diffplane[plot]))
    ax_diffred.append(fig.add_subplot(sub_gs0[1, 1], sharey=ax_diffplane[plot]))
    sub_gs1 = gs[plot, 3].subgridspec(2, 1, height_ratios=(1, 1), hspace=0.05)
    ax_fitness.append(fig.add_subplot(sub_gs1[0, 0]))
    ax_abundance.append(fig.add_subplot(sub_gs1[1, 0], sharex=ax_fitness[plot]))
    # add labels
    ax_diffplane[plot].set_xlabel("motility $d_A$", fontsize=12, color=colors[0])
    ax_diffplane[plot].set_ylabel("motility $d_I$", fontsize=12, color=colors[1])
    ax_red[plot].set_xlabel("space $x$", fontsize=12)
    ax_red[plot].set_ylabel("motility $d_I$", fontsize=12, color=colors[1])
    ax_blue[plot].set_xlabel("space $x$", fontsize=12)
    ax_blue[plot].set_ylabel("motility $d_A$", fontsize=12, color=colors[0])
    ax_fitness[plot].set_ylabel("fitness", fontsize=12)
    ax_fitness[plot].set_xticks([])
    ax_abundance[plot].set_ylabel("abundance", fontsize=12)
    ax_abundance[plot].set_xlabel("time $t$", fontsize=12)
    ax_abundance[plot].set_xticks([t_max[plot]-fitness_memory_time, t_max[plot]])

###
# RUN THE SIMULATION AND MAKE A SNAPSHOT
###
# Create a pattern object
env_prop = [pt.EnvProp() for plot in range(plot_num)]
spec_prop = [[pt.SpecProp(), pt.SpecProp()] for plot in range(plot_num)]
for plot in range(plot_num):
    if eco_forcing:
        env_prop[plot].int_fitness = pt.IntFit1(2, 0.62, 0.5)
        env_prop[plot].pos_num = 201
        env_prop[plot].initialize()
    else:
        env_prop[plot].int_fitness = pt.IntFit2(2.4, 8, 1, 1.2)
        env_prop[plot].pos_num = 201
        env_prop[plot].initialize()
    env_prop[plot].distant_mutations = distant_mutations
    env_prop[plot].diff_memory_time = t_max[plot]
    env_prop[plot].fitness_memory_time = fitness_memory_time
    for spec in range(2):
        spec_prop[plot][spec].diff_num = 11
        spec_prop[plot][spec].mut_rate = 1/150
        spec_prop[plot][spec].diff_num = 11
        spec_prop[plot][spec].initialize()
# Update pattern object up until t_max & make plot
for plot in range(plot_num):
    # Initialize the pattern
    np.random.seed(seed)
    pattern = pt.Pattern(env_prop[plot], spec_prop[plot], pt.init_pattern(env_prop[plot], spec_prop[plot], 1, None, 1, d_val[plot][0], d_val[plot][1]))
    # Update the pattern
    while pattern.time < t_max[plot]-env_prop[plot].time_step:
        pattern.diff_update()
        pattern.fitness_update()
        pattern.mutation_update()
        pattern.update_pattern_euler()
    # Plot the heatmaps
    pt.plot_heatmap_snapshot(pattern, [ax_blue[plot], ax_red[plot]], False)
    # Plot distribution of diffusivities
    pt.plot_diffusivity_distribution(pattern, [ax_diffplane[plot], ax_diffblue[plot], ax_diffred[plot]], False)
    # Plot expected diffusivity
    pt.plot_expected_diffusivity(pattern, [], ax_diffplane[plot], False, False)
    # Plot fitness and abundance
    pt.plot_fitness(pattern, 0, ax_fitness[plot], False, False)
    pt.plot_abundance(pattern, 0, ax_abundance[plot], False, False)
    if type == 0:
        if plot == 1:
            ax_fitness[plot].set_ylim([-10, 10])
            ax_fitness[plot].set_yticks([-15, 0, 15])
        else:
            ax_fitness[plot].set_ylim([-5, 5])
            ax_fitness[plot].set_yticks([-5, 0, 5])
        ax_abundance[plot].set_yticks([0, 20, 40])
        ax_abundance[plot].set_ylim([0,40])
    elif type == 1:
        ax_fitness[plot].set_ylim([-5, 5])
        ax_fitness[plot].set_yticks([-5, 0, 5])
        ax_abundance[plot].set_yticks([0, 20, 40])
        ax_abundance[plot].set_ylim([0, 40])
    else:
        if plot == 0:
            ax_fitness[plot].set_ylim([-15, 15])
            ax_fitness[plot].set_yticks([-15, 0, 15])
        else:
            ax_fitness[plot].set_ylim([-10, 10])
            ax_fitness[plot].set_yticks([-10, 0, 10])
        ax_abundance[plot].set_yticks([0, 5, 10])
        ax_abundance[plot].set_ylim([0, 10])
# Plot show and save
fig.savefig("Figures/Sketches/Fig4_ESS_Type"+str(type)+".svg")
plt.show()
