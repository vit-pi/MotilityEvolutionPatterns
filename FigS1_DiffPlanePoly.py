###
# IMPORTS
###
import matplotlib.pyplot as plt
import Pattern as pt

# No External parameters - background of Fig S1 is reproduced

###
# PLOT
###
# Create objects
env_prop = pt.EnvProp()
env_prop.int_fitness = pt.IntFit1(2,0.62,0.5) #pt.IntFit2(2.4,8,1,1.2)
spec_prop = [pt.SpecProp(), pt.SpecProp()]
spec_prop[0].diff_min = 1e-2
spec_prop[1].diff_min = 1
spec_prop[0].diff_max = 1
spec_prop[1].diff_max = 100
# Create figure
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(3*3.6, 2*3.3), constrained_layout=True)
# Make the plot
q_val = [0, 0.25, 0.5]
num = [1e3,1e4]
for row in range(2):
    for col in range(3):
        pt.plot_diffusivity_plane_poly(row,q_val[col],env_prop,spec_prop,axs[row][col],True, num[row])
fig.savefig("Figures/Sketches/Fig_DiffPlanePoly.svg")
plt.show()