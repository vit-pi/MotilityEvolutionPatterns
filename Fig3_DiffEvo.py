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
# External: By varying the truth value, Fig. 3a (true) and its SI complement (false) are reproduced
eco_forcing = True

# Internal
if eco_forcing:
    filename = "DiffEvo_PreyPred"
    t_max = 1800
    plot_times = [100, 300, 700, 1100]
else:
    filename = "DiffEvo_CoopDef"
    t_max = 1800
    plot_times = [100, 300, 800, 1250]
seed = 0
d0 = 0.1
d1 = 1

###
# CREATE AND FORMAT FIGURE
###
fig = plt.figure(figsize=(5.96, 5.96))
gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.4, hspace=0.4)
ax = fig.add_subplot(gs[:, :])

###
# RUN THE SIMULATION AND MAKE A SNAPSHOT
###
# Create a pattern object
env_prop = pt.EnvProp()
if eco_forcing:
    env_prop.int_fitness = pt.IntFit1(2,0.62,0.5)
    env_prop.pos_num = 101
else:
    env_prop.int_fitness = pt.IntFit2(2.4,8,1,1.2)
    env_prop.pos_num = 121
env_prop.initialize()
env_prop.diff_memory_time = t_max
spec_prop = [pt.SpecProp(), pt.SpecProp()]
for spec in range(2):
    spec_prop[spec].diff_num = 11
    spec_prop[spec].mut_rate = 5e-3
    spec_prop[spec].initialize()
pattern = pt.Pattern(env_prop, spec_prop, pt.init_pattern(env_prop, spec_prop, 1, seed, 1, d0, d1))
# Update pattern object up until t_max & make plot
snap = 0
while pattern.time < t_max:
    pattern.diff_update()
    pattern.mutation_update()
    if snap < len(plot_times) and pattern.time >= plot_times[snap]:
        print("Snapshot: " + str(snap))
        # Plot pattern heatmap to a separate file
        fig1, ax1 = plt.subplots(ncols=1, nrows=2, figsize=(1.5, 1), constrained_layout=True)
        pt.plot_cumulative_heatmap_snapshot(pattern, ax1, True)
        fig1.savefig("Figures/Sketches/Fig3_"+filename+"_Snap"+str(snap)+".svg")
        # Update snap
        snap += 1
    pattern.update_pattern_euler()
pattern.time = np.floor(pattern.time)
# Plot expected diffusivity
pt.plot_expected_diffusivity(pattern, plot_times, ax, True, True)
# Plot
fig.savefig("Figures/Sketches/Fig3_"+filename+".svg")
plt.show()
