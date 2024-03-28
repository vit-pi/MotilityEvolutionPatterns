###
# IMPORTS
###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Pattern as pt
import os
import datetime

###
# PARAMETERS
###
# External: By varying the truth values of the parameters below, Movies 1-5 are reproduced.
# Movie 1: distant_mutations=False, eco_forcing parameters=True, init_homog=False
# Movie 2: distant_mutations=False, eco_forcing parameters=True, init_homog=True
# Movie 3: distant_mutations=True, eco_forcing parameters=True, init_homog=False
# Movie 4: distant_mutations=False, eco_forcing parameters=False, init_homog=False

distant_mutations = False
eco_forcing = False
init_homog = True

# Internal
if eco_forcing:
    title = "Predator-Prey Model\n"
    filename = "PreyPred_"
else:
    title = "Cooperator-Defector Model\n"
    filename = "CoopDef_"
if distant_mutations:
    t_max = 1110
    plot_step = 10
    title += "Large-Effect Mutations\n"
    filename += "LongMut_"
else:
    t_max = 8010
    plot_step = 10
    title += "Small-Effect Mutations\n"
    filename += "ShortMut_"
seed = 0
if init_homog:
    t_max = 2010
    plot_step = 1
    d0 = 0.5
    d1 = 0.1
    filename += "InitHomog_"
else:
    d0 = 0.1
    d1 = 1
white_time = 10

###
# CREATE AND FORMAT FIGURE
###
fig = plt.figure(figsize=(6.25,6))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.5, hspace=0.35)
colors = ["#214478ff", "#aa0000ff"]    # [blue, red]
ax_blue = fig.add_subplot(gs[0, 0])
ax_blue.set_xlabel("space $x$", labelpad=-8)
ax_red = fig.add_subplot(gs[1, 0])
ax_red.set_xlabel("space $x$", labelpad=-8)
if eco_forcing:
    ax_blue.set_ylabel("prey motility $d_A$", labelpad=-7, color=colors[0])
    ax_red.set_ylabel("predator motility $d_I$", labelpad=-7, color=colors[1])
else:
    ax_blue.set_ylabel("cooperator motility $d_A$", labelpad=-7, color=colors[0])
    ax_red.set_ylabel("defector motility $d_I$", labelpad=-7, color=colors[1])
ax_fitness = fig.add_subplot(gs[1, 1])
ax_fitness.set_xlabel("time t")
ax_fitness.set_ylabel("$—$ total fitness $F_i$")
ax_abundance = ax_fitness.twinx()
ax_abundance.set_ylabel("$\\cdot\\cdot\\cdot$ species abundance")
if eco_forcing:
    if distant_mutations and not init_homog:
        ax_fitness.set_ylim(-10, 5)
        ax_fitness.set_yticks([-10, -5, 0, 5])
        ax_abundance.set_ylim(0, 30)
        ax_abundance.set_yticks([0, 10, 20, 30])
    else:
        ax_fitness.set_ylim(-60, 60)
        ax_fitness.set_yticks([-60, -30, 0, 30, 60])
        ax_abundance.set_ylim(0, 80)
        ax_abundance.set_yticks([0, 20, 40, 60, 80])
else:
    if distant_mutations and not init_homog:
        ax_fitness.set_ylim(-3, 1)
        ax_fitness.set_yticks([-3, -2, -1, 0, 1])
        ax_abundance.set_ylim(0, 20)
        ax_abundance.set_yticks([0, 5, 10, 15, 20])
    elif init_homog:
        ax_fitness.set_ylim(-10, 10)
        ax_fitness.set_yticks([-10, -5, 0, 5, 10])
        ax_abundance.set_ylim(0, 20)
        ax_abundance.set_yticks([0, 5, 10, 15, 20])
    else:
        ax_fitness.set_ylim(-20, 5)
        ax_fitness.set_yticks([-20, -15, -10, -5, 0, 5])
        ax_abundance.set_ylim(0, 20)
        ax_abundance.set_yticks([0, 5, 10, 15, 20])
sub_gs = gs[0, 1].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
ax_diffplane = fig.add_subplot(sub_gs[1, 0])
ax_diffplane.set_xlabel("motility $d_A$", labelpad=-8, color=colors[0])
ax_diffplane.set_ylabel("motility $d_I$", labelpad=-7, color=colors[1])
ax_diffblue = fig.add_subplot(sub_gs[0, 0], sharex=ax_diffplane)
ax_diffred = fig.add_subplot(sub_gs[1, 1], sharey=ax_diffplane)

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
env_prop.initialize()
spec_prop = [pt.SpecProp(), pt.SpecProp()]
pattern = pt.Pattern(env_prop, spec_prop, pt.init_pattern(env_prop, spec_prop, 1, seed, 1, d0, d1))
# Prepare folder
date_time = datetime.datetime.now()
folder_specifier = "_"+str(date_time.date().day)+"_"+str(date_time.date().month)+"_"+str(date_time.date().year)+"_"+str(date_time.time().hour) + "_" + str(date_time.time().minute)
folder_name = "Movies\\" + filename + folder_specifier
os.mkdir(folder_name)
# Update pattern object up until t_max & make plot
plot_index = 0
while pattern.time <= t_max:
    pattern.fitness_update()
    pattern.mutation_update()
    if pattern.time >= plot_step*plot_index:
        print("Snapshot: "+str(plot_index))
        # Plot pattern heatmaps
        pt.plot_heatmap_snapshot(pattern, [ax_blue, ax_red], False)
        # Plot distribution of diffusivities
        pt.plot_diffusivity_distribution(pattern, [ax_diffplane, ax_diffblue, ax_diffred], False)
        # Plot fitness
        pt.plot_fitness(pattern, white_time, ax_fitness, True, False)
        # Plot total abundance
        pt.plot_abundance(pattern, white_time, ax_abundance, False, False)
        # Format axes
        for i, ax in enumerate(fig.axes):
            ax.tick_params(axis='both', which='major', labelsize=10)
        # Update title, plot_index and save
        fig.suptitle(title + "$t = $" + "{:5.0f}".format(pattern.time))
        fig.savefig(folder_name + "\\" + filename + "-" + str(plot_index + 1) + ".png", dpi=300)
        plot_index += 1
        # Remove artists
        for i, ax in enumerate(fig.axes):
            for artist in ax.lines + ax.collections + ax.patches:
                artist.remove()
        for artist in ax_abundance.lines + ax_abundance.collections + ax_abundance.patches:
            artist.remove()
    pattern.update_pattern_euler()
