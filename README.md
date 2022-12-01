# DispersalEvolutionPatterns
This project provides the code to explore how the evolution of dispersal limits pattern formation [1]. It is instructive to firstly read [1], in order to understand this project. The code was written in Python.

## Code structure
### Pattern.py
This file contains the lower level objects and functions that are called by the rest of the files. Objects EnvProp and SpecProp store the information about the information of the environment where the studied organisms move (EnvProp) and about the properties of these organisms (SpecProp). The IntFit objects carry the information about fitness functions of each organism (see [1]) and are passed as an attribute to the EnvProp object. The EnvProp and SpecProp object are passed to the Pattern object (the main object), that holds the information about spatial distribution of organisms of all diffusivities at a particular time. This object contains methods to (1) transform between data structures that describe this distribution, (2) find the properties of this distribution, and (3) update this distribution in time.  The important update functions are:

1) update_pattern_euler(): used to update the distribution and time $t \to t + \Delta t$,
2) mutation_update(): used before every time-step of the Euler method above if mutatants of variable diffusivity are to be introduced into the population,
3) fitness_update(): used before every time-step of the Euler method and before mutation_update() to keep track of the total fitness of each organism,
4) fitness_update(): used before every time-step of the Euler method and before mutation_update() to keep track of the expected diffusivity of each organism.

To initialize the Pattern object, the initial pattern is required in addition to EnvProp and SpecProp objects. This can be achieved with the init_pattern function. To make a desired plot of the current state of the Pattern object, the plotting functions are used.

### Movie.py
This code reproduces all supplementary movies in [1]. The snapshots of the output movies are saved to new subfolders created in the Movies folder.

### FigX_file.py
These files reproduce various panels of the respective Figure X in [1]. Most FigX_file.py files contain External parameters specified at the beginning of each file, with instructions on which combination of these parameters reproduces which Figure (resp. Figure panel). The output preliminary figures are saved in the Figures/Sketches folder.

## Installation guide
(Tested on Windows 11 Home, Windows 11 Home, Intel(R) Core(TM) i7-8550U CPU, 16 GB RAM, Python and PyCharm installed. The estimated run time for producing a Figure panels or a Movie snapshots as in [1]: 1 second - 3 hours, depending on a particular task.)
1) Install Python (https://www.python.org/) and PyCharm (https://www.jetbrains.com/pycharm/).
2) Download this project and save it into a folder.
3) Open this folder with PyCharm.

## Guide to reproduce Figures and Movies of [1]
1) Open a file of interest (FigX_file.py or Movie.py).
2) Modify the "External parameters" in the file to reproduce the desired Figure or Movie in [1] (see instructions in each file).
3) Run the file.
4) Output file(s) appear in the Figures/Sketches folder (resp. Movies folder).

# References:
[1] V. Piskovsky, unpublished

