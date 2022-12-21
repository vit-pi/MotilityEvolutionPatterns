###
# IMPORTS
###
import matplotlib.pyplot as plt
import Pattern as pt

# No External parameters - background of Fig 1d is reproduced

###
# PLOT
###
# Create objects
env_prop = pt.EnvProp()
env_prop.int_fitness = pt.IntFit1(2,0.62,0.5) #pt.IntFit2(2.4,8,1,1.2)
spec_prop = [pt.SpecProp(), pt.SpecProp()]
spec_prop[0].diff_max = 1
spec_prop[1].diff_max = 1
# Create figure
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3.6, 3.3), constrained_layout=True)
# Make the plot
pt.plot_diffusivity_plane(env_prop,spec_prop,ax,True)
fig.savefig("Figures/Sketches/Fig1_DiffPlane.svg")
plt.show()