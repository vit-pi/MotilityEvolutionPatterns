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
# External: By varying the truth value, Fig. 3b-e (true) and its SI complement (false) are reproduced
eco_forcing = True

# Internal
seed = 0
d0_fix = 0.1
d1_fix = 1
d_var = [0, 1, 31]
before_t = 300
after_t = 200
spec_variable = 1
if eco_forcing:
    spec_variable = 0

###
# CREATE AND FORMAT FIGURE
###
fig = plt.figure(figsize=(8.95, 7.5))
gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=(2, 3), wspace=1, hspace=0.3)
ax_pip_0 = fig.add_subplot(gs[1, 0:3])
ax_pip_1 = fig.add_subplot(gs[1, 3:6])
ax_invader = fig.add_subplot(gs[0, 2:4])
ax_invader.set_ylabel("invaders", fontsize=12)
ax_invader.set_xlabel("time $t$", fontsize=12, labelpad=-7)
sub_gs0 = gs[0, 0:2].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
ax_diffplane0 = fig.add_subplot(sub_gs0[1, 0])
ax_diffplane0.set_xlabel("diffusivity $D_A$", fontsize=12, labelpad=-8)
ax_diffplane0.set_ylabel("diffusivity $D_I$", fontsize=12, labelpad=-7)
ax_diffblue0 = fig.add_subplot(sub_gs0[0, 0], sharex=ax_diffplane0)
ax_diffred0 = fig.add_subplot(sub_gs0[1, 1], sharey=ax_diffplane0)
sub_gs1 = gs[0, 4:6].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
ax_diffplane1 = fig.add_subplot(sub_gs1[1, 0])
ax_diffplane1.set_xlabel("diffusivity $D_A$", fontsize=12, labelpad=-8)
ax_diffplane1.set_ylabel("diffusivity $D_I$", fontsize=12, labelpad=-7)
ax_diffblue1 = fig.add_subplot(sub_gs1[0, 0], sharex=ax_diffplane1)
ax_diffred1 = fig.add_subplot(sub_gs1[1, 1], sharey=ax_diffplane1)
ax_diffplane = [ax_diffplane0, ax_diffplane1]
ax_diffblue = [ax_diffblue0, ax_diffblue1]
ax_diffred = [ax_diffred0, ax_diffred1]

###
# RUN THE SIMULATION AND MAKE A SNAPSHOT
###
# Prepare pattern objects
d_fix = [d0_fix, d1_fix]
colors = ["#214478ff", "#aa0000ff"]    # [blue, red]
env_prop = pt.EnvProp()
spec_prop = [pt.SpecProp(), pt.SpecProp()]
for spec_var in range(2):
    spec_fixed = 1 - spec_var
    # Make diff_plane plots
    pattern = pt.Pattern(env_prop, spec_prop, pt.init_pattern(env_prop, spec_prop, None, None, 1, d0_fix, d1_fix))
    if spec_fixed == 0:
        diff_invader = int(spec_prop[1].diff_num/3)
        #ax_diffplane[spec_var].axvline(x=d_fix[spec_fixed], color='black')
    else:
        diff_invader = int(spec_prop[0].diff_num/2)
        #ax_diffplane[spec_var].axhline(y=d_fix[spec_fixed], color="black")
    for pos in range(env_prop.pos_num):
        pattern.n[spec_var*spec_prop[0].diff_num*env_prop.pos_num+diff_invader*env_prop.pos_num+pos] += env_prop.int_fitness.fixed_point[0]/4
    pt.plot_diffusivity_distribution(pattern, [ax_diffplane[spec_var], ax_diffblue[spec_var], ax_diffred[spec_var]], False)
# Make invader evolution plots
# 1 = prepare properties
spec_var = spec_variable
spec_fixed = 1 - spec_var
if eco_forcing:
    env_prop.int_fitness = pt.IntFit1(2, 0.62, 0.5)
else:
    env_prop.int_fitness = pt.IntFit2(2.4, 8, 1, 1.2)
spec_prop[spec_fixed].diff_min = d_fix[spec_fixed]
spec_prop[spec_fixed].diff_max = d_fix[spec_fixed]
spec_prop[spec_fixed].initialize()
spec_prop[spec_var].diff_num = 2
for spec in range(2):
    spec_prop[spec].diff_discrete = True
# 2 = plot faster invader
spec_prop[spec_var].diff_min = d_fix[spec_var]
spec_prop[spec_var].diff_max = d_fix[spec_var]*2
spec_prop[spec_var].initialize()
inv_diff = 1
np.random.seed(seed)
pattern = pt.Pattern(env_prop, spec_prop, pt.init_pattern(env_prop, spec_prop, None, None, 2, d_fix[0], d_fix[1], before_t))
pt.plot_mut_pert(pattern, spec_var, inv_diff, after_t, True, seed+1, ax_invader, False)
# 3 = plot slower invader
spec_prop[spec_var].diff_min = d_fix[spec_var]/2
spec_prop[spec_var].diff_max = d_fix[spec_var]
spec_prop[spec_var].initialize()
inv_diff = 0
np.random.seed(seed)
pattern = pt.Pattern(env_prop, spec_prop, pt.init_pattern(env_prop, spec_prop, None, None, 2, d_fix[0], d_fix[1], before_t))
pt.plot_mut_pert(pattern, spec_var, inv_diff, after_t, False, seed+1, ax_invader, False)
# Make PIP plots
env_prop = pt.EnvProp()
spec_prop = [pt.SpecProp(), pt.SpecProp()]
if eco_forcing:
    # Make eco plots
    env_prop.int_fitness = pt.IntFit1(2, 0.62, 0.5)
    pt.plot_pip(env_prop, spec_prop, 1, d1_fix, d_var, before_t, after_t, ax_pip_0, True)
    pt.plot_pip(env_prop, spec_prop, 0, d0_fix, d_var, before_t, after_t, ax_pip_1, True)
else:
    # Make evo plots
    env_prop.int_fitness = pt.IntFit2(2.4, 8, 1, 1.2)
    pt.plot_pip(env_prop, spec_prop, 1, d1_fix, d_var, before_t, after_t, ax_pip_0, True)
    pt.plot_pip(env_prop, spec_prop, 0, d0_fix, d_var, before_t, after_t, ax_pip_1, True)
plt.show()
if eco_forcing:
    fig.savefig("Figures/Sketches/Fig3_PIPeco.svg")
else:
    fig.savefig("Figures/Sketches/Fig3_PIPevo.svg")