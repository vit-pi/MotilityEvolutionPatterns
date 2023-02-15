###
# IMPORTS
###
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import copy
import scipy.optimize as optimize

###
# PROPERTIES
###

# Contains properties of a particular species
class SpecProp:
    def __init__(self):
        self.diff_max = 1
        self.diff_min = 0
        self.diff_num = 11
        self.diff_step = 0
        self.diff_discrete = True
        self.mut_rate = 1/150
        self.mut_size = 1e-4
        self.survival_threshold = 5e-5
        self.initialize()

    def initialize(self):
        self.survival_threshold = self.mut_size/2
        if self.diff_min == self.diff_max:
            self.diff_num = 1
            self.diff_step = 1
        else:
            self.diff_step = (self.diff_max-self.diff_min)/(self.diff_num-1)


# Contains properties of an environment, choose 1 or 2 dimensions only!
class EnvProp:
    def __init__(self):
        self.pos_max = 100
        self.pos_num = 176
        self.time_step = 1e-1
        self.int_fitness = IntFit1(2,0.62,0.5) #IntFit2(2.4,8,1,1.2) #IntFit1(2,0.62,0.5)
        self.pos_step = 0
        self.pert_size = 1e-4
        self.distant_mutations = False    # True = mutations into all strategies, False=mutations only to nearest neighbours
        self.fitness_memory_time = 50
        self.diff_memory_time = 50
        self.initialize()

    def initialize(self):
        self.pos_step = self.pos_max / (self.pos_num - 1)


# Define the interaction function, use Alonso et. al. 2002 paper - predators, prey
class IntFit1:
    def __init__(self, beta, etha, epsilon):
        self.type = 0
        # parameters
        self.beta = beta
        self.etha = etha
        self.epsilon = epsilon
        # fixed point
        delta = self.epsilon*self.beta/self.etha
        n1 = 1-(self.etha/self.epsilon)*(delta-1)
        n2 = (delta-1)*n1
        self.fixed_point = [n1, n2]
        # values of the Jacobian at the fixed point
        self.j11 = -1+2*beta - 2*etha/epsilon - (epsilon*beta-etha)**2/(beta*epsilon**2)
        self.j12 = -etha**2/(epsilon**2*beta)
        self.j21 = (epsilon*beta-etha)**2/(epsilon*beta)
        self.j22 = etha*(etha-epsilon*beta)/(epsilon*beta)
        self.trJ = self.j11 + self.j22
        self.detJ = self.j11*self.j22-self.j12*self.j21
        # run various checks
        if self.etha <= self.epsilon*(self.beta-1) or self.etha >= self.epsilon*self.beta:
            print("Wrong interaction function. You have chosen parameters with non-physical homogeneous steady state!")
        if self.trJ >= 0 or self.detJ <= 0:
            print("Wrong interaction function. You have chosen parameters with unstable homogeneous steady state!")
        if self.j11 <= 0:
            print("Wrong interaction function. Turing instability cannot occur!")
        if beta < 0 or etha < 0 or epsilon < 0:
            print("Wrong interacting function. You have chosen non-physical values of the beta, etha, epsilon parameters.")

    # return fitness
    def fitness(self, n1, n2, spec):
        if spec == 0:
            return (1 - n1) - self.beta * n2 / (n2 + n1)
        else:
            return self.epsilon * self.beta * n1 / (n2 + n1) - self.etha

    # returns t/f whether or not Turing pattern occurs
    def turing(self, d0, d1):
        if self.trJ >= 0 or self.detJ <= 0 or self.etha <= self.epsilon*(self.beta-1) or self.etha >= self.epsilon*self.beta:
            return False
        elif self.j22*d0+self.j11*d1>2*np.sqrt(self.detJ):
            return True
        else:
            return True

    # return critical d1 for a given d0 so that for d1 above this value, a Turing instability occurs
    def critical_d1(self, d0):
        if 0 == self.j11:
            return 'NaN'
        elif 0 < self.j11:
            d1 = d0 * ((np.sqrt(self.detJ) + np.sqrt(-self.j12*self.j21)) / self.j11)**2
        else:
            d1 = d0 * ((np.sqrt(self.detJ) - np.sqrt(-self.j12*self.j21)) / self.j11)**2
        return d1

    # return critical d0 for a given d1 so that for d0 below this value, a Turing instability occurs
    def critical_d0(self, d1):
        if 0 == self.j22:
            return 'NaN'
        elif 0 < self.j22:
            d0 = d1 * ((np.sqrt(self.detJ) + np.sqrt(-self.j12*self.j21)) / self.j22)**2
        else:
            d0 = d1 * ((np.sqrt(self.detJ) - np.sqrt(-self.j12*self.j21)) / self.j22)**2
        return d0

# Define the interaction function, use Wakano et. al. 2009 paper - defectors, cooperators
class IntFit2:
    def __init__(self, r, n, b, d):
        self.type = 1
        # parameters
        self.r = r #2.5
        self.n = n #8
        self.b = b #1
        self.d = d #1.2
        # fixed point - hard to determine - need numerics
        self.fixed_point = optimize.fsolve(self.fixed_point_forcing, np.asarray([0.3, 0.3]))
        self.fixed_point_exists = False
        if self.fixed_point[0]>1e-9 and self.fixed_point[1]>1e-9:
            self.fixed_point_exists = True
        # values of the Jacobian at the fixed point
        n1 = self.fixed_point[0]
        n2 = self.fixed_point[1]
        y = n1 + n2
        x = 1 - y
        d1fd = r*n2*(1-(1-x**n)/(n*y))/y**2-r*n1/(y**3*n)*(n*x**(n-1)*y-1+x**n)
        d2fd = -r*n1*(1-(1-x**n)/(n*y))/y**2-r*n1/(y**3*n)*(n*x**(n-1)*y-1+x**n)
        dfdx = (r-1)*(n-1)*x**(n-2)-r/(y**2*n)*(-n*x**(n-1)*y+1-x**n)
        d1fc = d1fd+dfdx
        d2fc = d2fd+dfdx
        self.j11 = -n1*d/x+n1*x*d1fc
        self.j12 = -n1*d/x+n1*x*d2fc
        self.j21 = -n2*d/x+n2*x*d1fd
        self.j22 = -n2*d/x+n2*x*d2fd
        self.trJ = self.j11 + self.j22
        self.detJ = self.j11*self.j22-self.j12*self.j21
        # run various checks
        if not(self.fixed_point_exists):
            print("Wrong interaction function. You have chosen parameters with non-physical homogeneous steady state!")
        if self.trJ >= 0 or self.detJ <= 0:
            print("Wrong interaction function. You have chosen parameters with unstable homogeneous steady state!")
        if self.j11 <= 0:
            print("Wrong interaction function. Turing instability cannot occur!")
        if r < 0 or n < 0 or d < 0:
            print("Wrong interacting function. You have chosen non-physical values of the beta, etha, epsilon parameters.")

    # return value of supplementary function F(x)
    def fx(self,x):
        return 1+(self.r-1)*x**(self.n-1)-self.r/self.n*(1-x**self.n)/(1-x)

    # return fitness
    def fitness(self, n1, n2, spec):
        y = n1 + n2
        x = 1 - y
        fd = self.r * n1 * (1 - (1-x ** self.n) / (self.n * y)) / y
        if spec == 0:
            fc = fd - self.fx(x)
            return x*(fc+self.b)-self.d
        else:
            return x*(fd+self.b)-self.d

    # return fixed-point solver forcing
    def fixed_point_forcing(self, n):
        return np.asarray([self.fitness(n[0],n[1],0), self.fitness(n[0],n[1],1)])

    # returns t/f whether or not Turing pattern occurs
    def turing(self, d0, d1):
        if self.trJ >= 0 or self.detJ <= 0 or not(self.fixed_point_exists):
            return False
        elif self.j22*d0+self.j11*d1>2*np.sqrt(self.detJ):
            return True
        else:
            return True

    # return critical d1 for a given d0 so that ford d1 above this value, a Turing instability occurs
    def critical_d1(self, d0):
        if 0 == self.j11:
            return 'NaN'
        elif 0 < self.j11:
            d1 = d0 * ((np.sqrt(self.detJ) + np.sqrt(-self.j12*self.j21)) / self.j11)**2
        else:
            d1 = d0 * ((np.sqrt(self.detJ) - np.sqrt(-self.j12*self.j21)) / self.j11)**2
        return d1

    # return critical d0 for a given d1 so that for d0 below this value, a Turing instability occurs
    def critical_d0(self, d1):
        if 0 == self.j22:
            return 'NaN'
        elif 0 < self.j22:
            d0 = d1 * ((np.sqrt(self.detJ) + np.sqrt(-self.j12*self.j21)) / self.j22)**2
        else:
            d0 = d1 * ((np.sqrt(self.detJ) - np.sqrt(-self.j12*self.j21)) / self.j22)**2
        return d0



###
# PATTERN
###
# Main class that keeps the current pattern and includes methods to update this pattern in time
# Input: EnvProp, SpecProp (vector with two SpecProp objects)

class Pattern:

    ###
    # Properties and Pattern variables, Initialization
    ###
    # INPUT: spec_prop=[spec_prop1,spec_prop2], env_prop
    def __init__(self, env_prop, spec_prop, n_init):
        # properties:
        self.spec_prop = copy.deepcopy(spec_prop)
        self.env_prop = copy.deepcopy(env_prop)
        # pattern variables:
        vec_len = (self.spec_prop[0].diff_num + self.spec_prop[1].diff_num) * self.env_prop.pos_num
        self.n = np.zeros(vec_len)
        self.n_space_tot = np.zeros(2*self.env_prop.pos_num)
        self.time = 0
        # initialize pattern variables
        self.n = n_init
        self.update_n_space_tot()
        # fitness variable
        self.fitness_tot = np.zeros((2, round(self.env_prop.fitness_memory_time / self.env_prop.time_step)+1))
        self.fitness_tot[:] = np.nan
        self.exp_diff = np.zeros((2, round(self.env_prop.diff_memory_time / self.env_prop.time_step) + 1))
        self.exp_diff[:] = np.nan
        # adaptive dynamics parameters:
        self.active = np.ones(self.spec_prop[0].diff_num + self.spec_prop[1].diff_num, dtype=bool)
        self.next_mut_time = np.inf
        self.next_mut_spec = 0
        self.next_mut_time_memory = [[],[]]

    ###
    # Transformation between vectorised and array patterns
    ###
    # Import n[species][diff][pos] to self.n
    def array_to_vector(self, n_arr):
        i=0
        for spec in range(2):
            for diff in range(self.spec_prop[spec].diff_num):
                for pos in range(self.env_prop.pos_num):
                    self.n[i] = n_arr[spec][diff][pos]
                    i += 1

    # Read the value of vector self.n at a given indices of spec, diff, pos
    def vector_at_value(self, spec, diff, pos):
        return self.n[spec*self.spec_prop[0].diff_num*self.env_prop.pos_num+diff*self.env_prop.pos_num+pos]

    # Transform vector back to array
    def vector_to_array(self):
        i = 0
        n_arr = [np.zeros((self.spec_prop[0].diff_num,self.env_prop.pos_num)), np.zeros((self.spec_prop[1].diff_num,self.env_prop.pos_num))]
        for spec in range(2):
            for diff in range(self.spec_prop[spec].diff_num):
                for pos in range(self.env_prop.pos_num):
                    n_arr[spec][diff][pos] = self.n[i]
                    i += 1
        return n_arr
    ###
    # Measurements: expected diffusivity, etc
    ###

    # compute total spatial density of organisms by integrating over the diffusion space with trapezoidal rule
    def find_n_space_tot(self, pos, spec):
        space_tot = 0
        if self.spec_prop[spec].diff_discrete:
            for diff in range(self.spec_prop[spec].diff_num):
                space_tot += self.vector_at_value(spec, diff, pos)
        else:
            for diff in range(self.spec_prop[spec].diff_num-2):
                space_tot += self.vector_at_value(spec, diff+1, pos)
            space_tot += self.vector_at_value(spec, 0, pos)/2
            space_tot += self.vector_at_value(spec, self.spec_prop[spec].diff_num-1, pos)/2
            space_tot *= self.spec_prop[spec].diff_step
        return space_tot

    # compute total density of organisms of a given diffusion strategy by integration over the position space with trapezoidal rule
    def find_n_diff_tot(self, diff, spec):
        diff_tot = 0
        for pos in range(self.env_prop.pos_num-2):
            diff_tot += self.vector_at_value(spec, diff, pos+1)
        diff_tot += self.vector_at_value(spec, diff, 0)/2
        diff_tot += self.vector_at_value(spec, diff, self.env_prop.pos_num-1)/2
        diff_tot *= self.env_prop.pos_step
        return diff_tot

    # compute total number of organisms for a given species
    def find_n_tot(self, spec):
        tot = 0
        for pos in range(self.env_prop.pos_num-2):
            tot += self.find_n_space_tot(pos+1, spec)
        tot += self.find_n_space_tot(0, spec)/2
        tot += self.find_n_space_tot(self.env_prop.pos_num-1, spec)/2
        tot *= self.env_prop.pos_step
        return tot

    # compute int_0^diff_max \dd diff * n[spec][diff][pos] with the trapezoidal rule
    def find_diff_space_tot(self, pos, spec):
        diff_tot = 0
        if self.spec_prop[spec].diff_discrete:
            for diff in range(self.spec_prop[spec].diff_num):
                diff_tot += (self.spec_prop[spec].diff_step*diff+self.spec_prop[spec].diff_min)*self.vector_at_value(spec, diff, pos)
        else:
            for diff in range(self.spec_prop[spec].diff_num-2):
                diff_tot += (self.spec_prop[spec].diff_step*diff+self.spec_prop[spec].diff_min)*self.vector_at_value(spec, diff+1, pos)
            diff_tot += self.spec_prop[spec].diff_min*self.vector_at_value(spec, 0, pos)/2
            diff_tot += self.spec_prop[spec].diff_max*self.vector_at_value(spec, self.spec_prop[spec].diff_num-1, pos)/2
            diff_tot *= self.spec_prop[spec].diff_step
        return diff_tot

    # compute int_0^pos_max \dd pos * diff_space_tot[spec][pos] with the trapezoidal rule
    def find_diff_tot(self, spec):
        tot_diff = 0
        for pos in range(self.env_prop.pos_num-2):
            tot_diff += self.find_diff_space_tot(pos+1, spec)
        tot_diff += self.find_diff_space_tot(0, spec)/2
        tot_diff += self.find_diff_space_tot(self.env_prop.pos_num-1, spec)/2
        tot_diff *= self.env_prop.pos_step
        return tot_diff

    # find expected diffusivity of a given species
    def find_exp_diff(self, spec):
        tot_num = self.find_n_tot(spec)
        if tot_num > 0:
            return self.find_diff_tot(spec)/tot_num
        else:
            return None

    # find peaks in n_diff_tot together with their weights
    # output: n_diff_tot histogram (vector), list of peaks [diff index, expected diff in the neighbourhood, weight]
    def find_n_diff_peaks(self, spec):
        # prepare vectors
        n_diff = np.zeros(self.spec_prop[spec].diff_num)
        peaks = []
        # find n_diff_tot
        for diff in range(self.spec_prop[spec].diff_num):
            n_diff[diff] = self.find_n_diff_tot(diff, spec)
        # for each diff, determine whether n_diff has a maximum (peak), and if so compute the 3-point approximate expected diff
        for diff in range(self.spec_prop[spec].diff_num):
            peak = True
            tot_diff = n_diff[diff]
            exp_diff = self.spec_prop[spec].diff_step*diff*n_diff[diff]
            # left point exists
            if diff > 0:
                if n_diff[diff - 1] >= n_diff[diff]:
                    peak = False
                else:
                    tot_diff += n_diff[diff-1]
                    exp_diff += self.spec_prop[spec].diff_step * (diff-1) * n_diff[diff-1]
            # right point exists
            if diff < self.spec_prop[spec].diff_num - 1:
                if n_diff[diff + 1] >= n_diff[diff]:
                    peak = False
                else:
                    tot_diff += n_diff[diff+1]
                    exp_diff += self.spec_prop[spec].diff_step * (diff+1) * n_diff[diff+1]
            # save a peak
            if peak:
                if tot_diff > 0:
                    exp_diff = self.spec_prop[spec].diff_min + exp_diff/tot_diff
                else:
                    exp_diff = np.nan
                peaks.append([diff, exp_diff])
        # find total weight of all peaks
        total_weight = 0
        for peak in peaks:
            total_weight += n_diff[peak[0]]
        # add the relative weight of each peak
        for peak in peaks:
            if total_weight > 0:
                peak.append(n_diff[peak[0]]/total_weight)
            else:
                peak.append(np.nan)
        # create the histogram vector
        tot = self.find_n_tot(spec)
        if tot > 0:
            n_diff = n_diff/tot
        else:
            n_diff = np.zeros(self.spec_prop[spec].diff_num)
        return n_diff, peaks

    # find total fitness of a given species
    def find_fitness_tot(self, spec):
        fit_tot = 0
        for pos in range(self.env_prop.pos_num - 2):
            fit_tot += self.env_prop.int_fitness.fitness(self.n_space_tot[pos+1], self.n_space_tot[pos+self.env_prop.pos_num+1], spec)
        fit_tot += self.env_prop.int_fitness.fitness(self.n_space_tot[0], self.n_space_tot[self.env_prop.pos_num], spec) / 2
        fit_tot += self.env_prop.int_fitness.fitness(self.n_space_tot[self.env_prop.pos_num - 1], self.n_space_tot[2*self.env_prop.pos_num-1], spec) / 2
        fit_tot *= self.env_prop.pos_step
        return fit_tot

    # compute forcing for a given n[spec][diff][pos] variable
    def find_forcing(self, spec, diff, pos):
        force = 0
        if pos>0:
            force += (self.spec_prop[spec].diff_min+self.spec_prop[spec].diff_step*diff)*(self.vector_at_value(spec, diff, pos-1)-self.vector_at_value(spec, diff, pos))/self.env_prop.pos_step**2
        if pos<self.env_prop.pos_num-1:
            force += (self.spec_prop[spec].diff_min+self.spec_prop[spec].diff_step*diff)*(self.vector_at_value(spec, diff, pos+1) - self.vector_at_value(spec, diff, pos))/self.env_prop.pos_step**2
        force += self.vector_at_value(spec, diff, pos)*self.env_prop.int_fitness.fitness(self.n_space_tot[pos],self.n_space_tot[pos+self.env_prop.pos_num],spec)
        return force

    # update forcing vector
    def find_forcing_vector(self):
        vec_len = (self.spec_prop[0].diff_num + self.spec_prop[1].diff_num) * self.env_prop.pos_num
        forcing = np.zeros(vec_len)
        i = 0
        for spec in range(2):
            for diff in range(self.spec_prop[spec].diff_num):
                for pos in range(self.env_prop.pos_num):
                    forcing[i] = self.find_forcing(spec, diff, pos)
                    i += 1
        return forcing

    ###
    # Time-evolution of pattern variables
    ###

    # update self.n_space_tot
    def update_n_space_tot(self):
        for spec in range(2):
            for pos in range(self.env_prop.pos_num):
                self.n_space_tot[spec*self.env_prop.pos_num+pos] = self.find_n_space_tot(pos, spec)

    # update mutations
    def mutation_update(self):
        # kill strategies that are lost, abundance below survival_threshold*pos_max
        if self.time == 0 or self.time > self.next_mut_time:
            spec = self.next_mut_spec
            #for spec in range(2):
            for diff in range(self.spec_prop[spec].diff_num):
                i = spec * self.spec_prop[0].diff_num + diff
                if self.find_n_diff_tot(diff, spec) <= self.spec_prop[spec].survival_threshold * self.env_prop.pos_max:
                    for pos in range(self.env_prop.pos_num):
                        self.n[i * self.env_prop.pos_num + pos] = 0
                    self.active[i] = False
        # introduce mutations
        if self.time > self.next_mut_time:
            # introduce short-range mutations of a given species if there is a neigbouring active strategy
            if self.env_prop.distant_mutations:
                spec = self.next_mut_spec
                for diff in range(self.spec_prop[spec].diff_num):
                    i = spec*self.spec_prop[0].diff_num+diff
                    # am I active?
                    if not(self.active[i]):
                        self.active[i] = True
                        for pos in range(self.env_prop.pos_num):
                            self.n[i*self.env_prop.pos_num+pos] = self.spec_prop[spec].mut_size*np.random.rand()
            # introduce long-range mutations of a given species if the strategy is inactive
            else:
                spec = self.next_mut_spec
                old_active = copy.deepcopy(self.active)
                for diff in range(self.spec_prop[spec].diff_num):
                    i = spec*self.spec_prop[0].diff_num+diff
                    # am I active?
                    if not(old_active[i]):
                        # do I have an active neighbour?
                        if (diff > 0 and old_active[i-1]) or (diff<self.spec_prop[spec].diff_num-1 and old_active[i+1]):
                            self.active[i] = True
                            for pos in range(self.env_prop.pos_num):
                                self.n[i*self.env_prop.pos_num+pos] = self.spec_prop[spec].mut_size*np.random.rand()
        # update next_mut_time and next_mut_spec
        if self.time > self.next_mut_time or self.time == 0:
            total_mut_rate = self.spec_prop[0].mut_rate+self.spec_prop[0].mut_rate
            if total_mut_rate == 0:
                self.next_mut_time = np.inf
            else:
                self.next_mut_time = self.time-np.log(np.random.rand())/total_mut_rate
                if np.random.rand() < self.spec_prop[0].mut_rate/total_mut_rate:
                    self.next_mut_spec = 0
                else:
                    self.next_mut_spec = 1
                self.next_mut_time_memory[self.next_mut_spec].append(self.next_mut_time)

    # update pattern with Euler method
    def update_pattern_euler(self):
        # update self.n
        i=0
        for spec in range(2):
            for diff in range(self.spec_prop[spec].diff_num):
                if self.active[spec*self.spec_prop[0].diff_num+diff]:
                    for pos in range(self.env_prop.pos_num):
                        self.n[i] = self.n[i]+self.env_prop.time_step*self.find_forcing(spec, diff, pos)
                        i += 1
                else:
                    i += self.env_prop.pos_num
        # update self.n_space_tot
        self.update_n_space_tot()
        # update self.time
        self.time += self.env_prop.time_step

    # fitness update
    def fitness_update(self):
        index = round(self.time/self.env_prop.time_step) % (round(self.env_prop.fitness_memory_time/self.env_prop.time_step)+1)
        for spec in range(2):
            self.fitness_tot[spec][index] = self.find_fitness_tot(spec)

    # expected diffusivity update
    def diff_update(self):
        index = round(self.time/self.env_prop.time_step) % (round(self.env_prop.diff_memory_time/self.env_prop.time_step)+1)
        for spec in range(2):
            self.exp_diff[spec][index] = self.find_exp_diff(spec)

###
# PATTERN INITIALIZER
###
# Function that outputs appropriate initial pattern
# Input: same env_prop and spec_prop as specified for the pattern, pert_type, seed, type, specifiers
# type = 0 (homogeneous in space and diffusion space)
# type = 1 (homogeneous only in space) -> need [d0,d1] at which the pattern is fixed
# type = 2 (Turing pattern) -> need [d0,d1,t_max] for which the Turing pattern is simulated
# perturbation = None or a given type as specified below
def init_pattern(env_prop, spec_prop, pert_type, seed, type, *specifiers):
    # initial vector
    vec_len = (spec_prop[0].diff_num + spec_prop[1].diff_num) * env_prop.pos_num
    n = np.zeros(vec_len)
    # determine the closes diff indices that correspond to d0, d1
    diff_0 = 0
    diff_1 = 0
    if type == 1 or type == 2:
        if specifiers[0] < spec_prop[0].diff_min:
            diff_0 = 0
        elif specifiers[0] > spec_prop[0].diff_max:
            diff_0 = spec_prop[0].num - 1
        else:
            diff_0 = int((specifiers[0] - spec_prop[0].diff_min) / spec_prop[0].diff_step)
        if specifiers[1] < spec_prop[1].diff_min:
            diff_1 = 0
        elif specifiers[1] > spec_prop[1].diff_max:
            diff_1 = spec_prop[1].num - 1
        else:
            diff_1 = int((specifiers[1] - spec_prop[1].diff_min) / spec_prop[1].diff_step)
    # add the state of a given type
    if type == 0:
        # add the homogeneous state
        for i in range(vec_len):
            if i < spec_prop[0].diff_num * env_prop.pos_num:
                if spec_prop[0].diff_min == spec_prop[0].diff_max:
                    n[i] = env_prop.int_fitness.fixed_point[0]
                else:
                    n[i] = env_prop.int_fitness.fixed_point[0] / (spec_prop[0].diff_max - spec_prop[0].diff_min)
            else:
                if spec_prop[1].diff_min == spec_prop[1].diff_max:
                    n[i] = env_prop.int_fitness.fixed_point[1]
                else:
                    n[i] = env_prop.int_fitness.fixed_point[1] / (spec_prop[1].diff_max - spec_prop[1].diff_min)
    if type == 1:
        # add the homogeneous state
        for pos in range(env_prop.pos_num):
            n[diff_0*env_prop.pos_num+pos] += env_prop.int_fitness.fixed_point[0]
            n[spec_prop[0].diff_num*env_prop.pos_num+diff_1*env_prop.pos_num+pos] += env_prop.int_fitness.fixed_point[1]
    if type == 2:
        # create a pattern with same properties, but single diffusion strategies
        # 1 = modify the properties
        spec_propX = copy.deepcopy(spec_prop)
        spec_propX[0].diff_max = specifiers[0]
        spec_propX[0].diff_min = specifiers[0]
        spec_propX[0].initialize()
        spec_propX[1].diff_max = specifiers[1]
        spec_propX[1].diff_min = specifiers[1]
        spec_propX[1].initialize()
        # 2 = generate the initial pattern
        pattern = Pattern(env_prop, spec_propX, init_pattern(env_prop, spec_propX, 0, None, 1, specifiers[0], specifiers[1], specifiers[2]))
        # 3 = update the pattern up until time t
        while pattern.time < specifiers[2]:
            pattern.update_pattern_euler()
        # use the created pattern as the initial condition
        for pos in range(env_prop.pos_num):
            n[diff_0 * env_prop.pos_num + pos] += pattern.n[pos]
            n[spec_prop[0].diff_num*env_prop.pos_num+diff_1*env_prop.pos_num+pos] += pattern.n[env_prop.pos_num + pos]
    n = perturbation(env_prop, spec_prop, seed, n, pert_type, [diff_0, diff_1])
    return n

# Function that outputs appropriate perturbation
# Input: env_prop, spec_prop, seed (if specified it enforces the precise perturbation; if not new random perturbation is chosen), n, type, diff_0, diff_1
# type = 0 (homogeneous perturbation)
# type = 1 (perturbation at diff indices [diff_0,diff_1] only!)
# type = 2 (perturbation everywhere but at diff indices [diff_0,diff_1]!)
def perturbation(env_prop, spec_prop, seed, n, type, diffs):
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    if type == 0:
        for i in range(n.size):
            if n[i]<=env_prop.pert_size/2:
                n[i] = env_prop.pert_size*np.random.rand()
            else:
                n[i] = n[i]*(1+env_prop.pert_size*(np.random.rand()-0.5))
    elif type == 1:
        for spec in range(2):
            for pos in range(env_prop.pos_num):
                i = spec*spec_prop[0].diff_num*env_prop.pos_num+diffs[spec]*env_prop.pos_num+pos
                if n[i] <= env_prop.pert_size / 2:
                    n[i] = env_prop.pert_size * np.random.rand()
                else:
                    n[i] = n[i] * (1 + env_prop.pert_size * (np.random.rand() - 0.5))
    elif type == 2:
        for spec in range(2):
            for diff in range(spec_prop[spec].diff_num):
                if diff != diffs[spec]:
                    for pos in range(env_prop.pos_num):
                        i = spec * spec_prop[0].diff_num * env_prop.pos_num + diff * env_prop.pos_num + pos
                        if n[i] <= env_prop.pert_size / 2:
                            n[i] = env_prop.pert_size * np.random.rand()
                        else:
                            n[i] = n[i] * (1 + env_prop.pert_size * (np.random.rand() - 0.5))
    return n

###
# PLOTTING FUNCTIONS
###
# Functions that create a particular type of plot

# Plot a heatmap snapshot of the current state of the pattern
# Input: pattern, axs=[ax0, ax1] (one axis for each species)
def plot_heatmap_snapshot(pattern, axs, labels):
    colors = ["#214478ff", "#aa0000ff"]  # [blue, red]
    n = pattern.vector_to_array()
    images = []
    for spec in range(2):
        # prepare axes
        pos_axis = np.linspace(0, pattern.env_prop.pos_max, num=pattern.env_prop.pos_num+1)
        if pattern.spec_prop[spec].diff_min == pattern.spec_prop[spec].diff_max:
            diff_axis = np.linspace(pattern.spec_prop[spec].diff_min, pattern.spec_prop[spec].diff_max+1,num=2)
        else:
            diff_axis = np.linspace(pattern.spec_prop[spec].diff_min, pattern.spec_prop[spec].diff_max, num=pattern.spec_prop[spec].diff_num+1)
        # prepare colormap
        my_cmap = [cm.get_cmap('Blues').copy(), cm.get_cmap('Reds').copy()]
        # make plot and add features
        if pattern.env_prop.int_fitness.type == 0:
            vmax = [0.5,0.2]
        elif pattern.env_prop.int_fitness.type == 1:
            vmax = [0.14, 0.12]
        im = axs[spec].pcolormesh(pos_axis, diff_axis, n[spec], cmap=my_cmap[spec], vmin=0, vmax=vmax[spec])
        images.append(im)
        # modify ticks and tick_labels
        axs[spec].set_xticks([0, pattern.env_prop.pos_max])
        axs[spec].tick_params(axis='both', which='major', labelsize=12)
        if pattern.spec_prop[spec].diff_min == pattern.spec_prop[spec].diff_max:
            axs[spec].set_yticks([])
            pad = 0
        else:
            axs[spec].set_yticks([pattern.spec_prop[spec].diff_min+pattern.spec_prop[spec].diff_step/2, pattern.spec_prop[spec].diff_max-pattern.spec_prop[spec].diff_step/2], fontsize=12)
            axs[spec].set_yticklabels([pattern.spec_prop[spec].diff_min, pattern.spec_prop[spec].diff_max])
            pad = 1
        # labels
        if labels:
            axs[spec].set_xlabel("space $x$", fontsize=12, labelpad=-8)
            if spec == 0:
                axs[spec].set_ylabel("motility $d_A$", fontsize=12, labelpad=-7*pad, color=colors[0])
            else:
                axs[spec].set_ylabel("motility $d_I$", fontsize=12, labelpad=-7*pad, color=colors[1])
    return images

# Plot a cummulative heatmap snapshot of the current state of the pattern (i.e. integrate over diffusivities)
# Input: pattern, axs=[ax0, ax1] (one axis for each species)
def plot_cumulative_heatmap_snapshot(pattern, axs, labels):
    # prepare pattern vector
    n_space_tot = np.zeros((2,pattern.env_prop.pos_num))
    for spec in range(2):
        for pos in range(pattern.env_prop.pos_num):
            n_space_tot[spec][pos] = pattern.n_space_tot[spec*pattern.env_prop.pos_num+pos]
    images = []
    for spec in range(2):
        # prepare axes
        pos_axis = np.linspace(0, pattern.env_prop.pos_max, num=pattern.env_prop.pos_num+1)
        diff_axis = np.linspace(pattern.spec_prop[spec].diff_min, pattern.spec_prop[spec].diff_min+1,num=2)
        # prepare colormap
        my_cmap = [cm.get_cmap('Blues').copy(), cm.get_cmap('Reds').copy()]
        # make plot and add features
        if pattern.env_prop.int_fitness.type == 0:
            vmax = [0.5,0.2]
        elif pattern.env_prop.int_fitness.type == 1:
            vmax = [0.14, 0.12]
        im = axs[spec].pcolormesh(pos_axis, diff_axis, [n_space_tot[spec]], cmap=my_cmap[spec], vmin=0, vmax=vmax[spec])
        images.append(im)
        # modify ticks and tick_labels
        axs[spec].set_xticks([])
        axs[spec].set_yticks([])
        # labels
        if labels:
            axs[1].set_xlabel("space $x$", fontsize=12)
    return images

# Plot the plane of diffusivities
# Input: env_prop, spec_prop, ax
def plot_diffusivity_plane(env_prop, spec_prop, ax, labels):
    # prepare diffusivity axes
    colors = ["#214478ff", "#aa0000ff"]  # [blue, red]
    num = 100
    d0 = np.linspace(spec_prop[0].diff_min, 2*spec_prop[0].diff_max, num=num)
    d1 = np.zeros(num)
    # compute critical d1 for each d0
    for i in range(num):
        d1[i] = env_prop.int_fitness.critical_d1(d0[i])
        if np.isnan(d1[i]):
            d1[i] = 2*spec_prop[1].diff_max
    # make the plot
    ax.plot(d0, d1, color='black', linewidth=2, linestyle="--")
    # fill the regions with colour
    ax.fill_between(d0, d1, color='#ccccccff')
    ax.fill_between(d0, d1, np.ones(num)*spec_prop[1].diff_max*2, color='#999999ff')
    # set x,y limits
    ax.set_xlim(spec_prop[0].diff_min, spec_prop[0].diff_max)
    ax.set_ylim(spec_prop[1].diff_min, spec_prop[1].diff_max)
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # put labels
    if labels:
        ax.set_xlabel("motility $d_A$", fontsize=12, color=colors[0])
        ax.set_ylabel("motility $d_I$", fontsize=12, color=colors[1])

# Plot diffusivity distribution
# Input: pattern, axs = [ax_diffplane, ax_diffblue, ax_diffred], labels
def plot_diffusivity_distribution(pattern, axs, labels):
    # get the peak and histogram data
    peaks = [pattern.find_n_diff_peaks(0), pattern.find_n_diff_peaks(1)]
    # plot the diffusivity plane
    plot_diffusivity_plane(pattern.env_prop, pattern.spec_prop, axs[0], False)
    # plot the peaks in the ax_diffplane
    for peak0 in peaks[0][1]:
        for peak1 in peaks[1][1]:
            axs[0].plot(peak0[1], peak1[1], marker="o", markersize=10*np.sqrt(peak0[2]*peak1[2]), color='black', zorder=10, clip_on=False)
    # plot histograms
    colors = ["#214478ff", "#aa0000ff"]    # [blue, red]
    for i in range(2):
        diff_axis = np.linspace(pattern.spec_prop[i].diff_min, pattern.spec_prop[i].diff_max, num=pattern.spec_prop[i].diff_num)
        if i == 0:
            axs[i+1].bar(diff_axis, peaks[i][0], color=colors[i], width=pattern.spec_prop[i].diff_step, zorder=10, clip_on=False)
        else:
            axs[i+1].barh(diff_axis, peaks[i][0], color=colors[i], height=pattern.spec_prop[i].diff_step, zorder=10, clip_on=False)
    # set limits
    axs[1].set_xlim(0,pattern.spec_prop[0].diff_max+pattern.spec_prop[0].diff_step)
    axs[1].set_ylim(0,1.1)
    axs[2].set_xlim(0,1.1)
    axs[2].set_ylim(0,pattern.spec_prop[1].diff_max+pattern.spec_prop[1].diff_step)
    # set ticks
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].set_xticks([pattern.spec_prop[0].diff_min, pattern.spec_prop[0].diff_max])
    axs[0].set_yticks([pattern.spec_prop[1].diff_min, pattern.spec_prop[1].diff_max])
    axs[1].tick_params(labelbottom=False, tick1On=False)
    axs[1].set_yticks([])
    axs[2].tick_params(labelleft=False, tick1On=False)
    axs[2].set_xticks([])
    # include labels
    if labels:
        axs[0].set_xlabel("motility $d_A$", fontsize=12, labelpad=-8, color=colors[0])
        axs[0].set_ylabel("motility $d_I$", fontsize=12, labelpad=-7, color=colors[1])

# Plot evolution of fitness in time
# Input: pattern, white_time (extra time with no plot), ax, labels
def plot_fitness(pattern, white_time, ax, mutation_times, labels):
    # define colors
    colors = ["#214478ff", "#aa0000ff"]    # [blue, red]
    # prepare the times and fitness data
    fitness_num = np.size(pattern.fitness_tot[0])
    times = np.linspace(pattern.time-pattern.env_prop.fitness_memory_time, pattern.time, num=fitness_num)
    index = round(pattern.time / pattern.env_prop.time_step) % fitness_num
    fitness = np.roll(pattern.fitness_tot, -(index+1),axis=1)
    # plot 0 axis
    ax.axhline(y=0, color='black')
    # draw current time line
    if white_time>0:
        ax.axvline(x=times[-1], color='grey', linestyle="--")
    # make the plot
    for i in range(2):
        ax.plot(times, fitness[i], color=colors[i])
    # plot mutation times
    if mutation_times:
        for spec in range(2):
            for mut_time in reversed(pattern.next_mut_time_memory[spec]):
                if mut_time <= pattern.time:
                    ax.axvline(x=mut_time, color=colors[spec], linestyle='--', linewidth=0.8)
                if mut_time < pattern.time-pattern.env_prop.fitness_memory_time:
                    break
    # set limits
    ax.set_xlim(times[0], times[-1]+white_time)
    ax.set_xlim(times[0], times[-1]+white_time)
    # add labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    if labels:
        ax.set_xlabel("time t", fontsize=12)
        ax.set_ylabel("total fitness $F_i$", fontsize=12)

# Plot evolution of diffusivity in time
# Input: pattern, ax, labels, background (diffusivity plane)
def plot_expected_diffusivity(pattern, plot_times, ax, background, labels):
    # define colors
    colors = ["#214478ff", "#aa0000ff"]    # [blue, red]
    # prepare the times and fitness data
    exp_diff_num = np.size(pattern.exp_diff[0])
    times = np.linspace(pattern.time-pattern.env_prop.diff_memory_time, pattern.time, num=exp_diff_num)
    index = round(pattern.time / pattern.env_prop.time_step) % exp_diff_num
    exp_diff = np.roll(pattern.exp_diff, -(index+1), axis=1)
    # plot diffusivity plane
    if background:
        plot_diffusivity_plane(pattern.env_prop, pattern.spec_prop, ax, False)
    # plot diffusivity trajectory
    ax.plot(exp_diff[0], exp_diff[1], color="#d4aa00ff", linewidth=3, zorder=10, clip_on=False)
    # mark diffusivity at specified times
    for time in plot_times:
        i = round(time / pattern.env_prop.time_step) % exp_diff_num
        ax.plot(exp_diff[0][i], exp_diff[1][i], marker="o", markersize=10, color="#d4aa00ff", zorder=10, clip_on=False)
    # set ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks([pattern.spec_prop[0].diff_min, pattern.spec_prop[0].diff_max])
    ax.set_yticks([pattern.spec_prop[1].diff_min, pattern.spec_prop[1].diff_max])
    # set x,y limits
    ax.set_xlim(pattern.spec_prop[0].diff_min, 1.1*pattern.spec_prop[0].diff_max)
    ax.set_ylim(pattern.spec_prop[1].diff_min, 1.1*pattern.spec_prop[1].diff_max)
    # include labels
    if labels:
        ax.set_xlabel("motility $d_A$", fontsize=12, labelpad=-8, color=colors[0])
        ax.set_ylabel("motility $d_I$", fontsize=12, labelpad=-7, color=colors[1])


# Measure invasion exponent
# Input: initialized pattern (1 and 2 diffusivity values for each species),
#        inv_spec (species with 2 diffusivities), inv_diff (diffusibility corresponding to invasion - 0,1)
#        after_t (time after invasion)
# Output: -1 (negative exponent), 0 (neutral exponent), 1 (positive exponent)
def find_invasion_exponent(pattern, inv_spec, inv_diff, after_t):
    # introduce small proportion of invaders
    for pos in range(pattern.env_prop.pos_num):
        i = inv_spec * pattern.spec_prop[0].diff_num * pattern.env_prop.pos_num + inv_diff * pattern.env_prop.pos_num + pos
        pattern.n[i] = pattern.spec_prop[inv_spec].mut_size * np.random.rand()
    m_0 = pattern.find_n_diff_tot(inv_diff, inv_spec)
    # continue evolution for time after_t
    while pattern.time<after_t:
        pattern.update_pattern_euler()
    m_t = pattern.find_n_diff_tot(inv_diff, inv_spec)
    if m_t > m_0:
        return 1
    elif m_t < m_0:
        return -1
    else:
        return 0
    #estimate on the value of invasion exponenet is np.log(m_t/m_0)/after_t

# Plot PIP
# Input: evn_prop, spec_prop, spec_fix, d_fix (species spec_fix with fixed diffusivity d_fix), d_var=[d_min,d_max,d_num]
#        (species with variable diffusivity), before_t (time before invasion), after_t (time after invasion), ax, labels
def plot_pip(env_prop, spec_prop, spec_fixed, d_fix, d_var, before_t, after_t, ax, labels):
    # modify species properties
    spec_variable = 1 - spec_fixed
    spec_prop[spec_fixed].diff_min = d_fix
    spec_prop[spec_fixed].diff_max = d_fix
    spec_prop[spec_fixed].initialize()
    for spec in range(2):
        spec_prop[spec].diff_discrete = True
    # create axis variable
    diff_axis = np.linspace(d_var[0], d_var[1], num=d_var[2])
    inv_exponents = np.zeros((d_var[2]-1, d_var[2]-1))
    # create pattern variables (current_n, homog_n, pattern_state):
    # 0 = evolve perturbation near homogeneous state, 1 = evolve previous pattern, 2 = use homogeneous state
    current_n = None
    pattern_state = 0
    steady_n = np.zeros(2*env_prop.pos_num)
    for i in range(2*env_prop.pos_num):
        if i <  env_prop.pos_num:
            steady_n[i] = env_prop.int_fitness.fixed_point[0]
        else:
            steady_n[i] = env_prop.int_fitness.fixed_point[1]
    # for each pair, get the invasion exponent
    for diff_res in range(d_var[2]-1):
        # start from largest diffusivity of an inhibitor - species 1
        if spec_variable == 1:
            diff_res = d_var[2]-2-diff_res
        # if the Turing pattern does not form, use homogeneous state
        if not env_prop.int_fitness.turing(d_fix, diff_axis[diff_res]):
            pattern_state = 2
        # find the stable pattern
        if pattern_state == 0:
            # 1 = prepare species properties
            spec_prop[spec_variable].diff_min = diff_axis[diff_res]
            spec_prop[spec_variable].diff_max = diff_axis[diff_res]
            spec_prop[spec_variable].initialize()
            # 2 = create pattern
            if spec_fixed == 0:
                pattern = Pattern(env_prop, spec_prop,
                                  init_pattern(env_prop, spec_prop, None, None, 2, d_fix, diff_axis[diff_res], before_t))
            else:
                pattern = Pattern(env_prop, spec_prop,
                                  init_pattern(env_prop, spec_prop, None, None, 2, diff_axis[diff_res], d_fix, before_t))
            # 3 = update pattern variables
            current_n = pattern.n
            pattern_state = 1
        if pattern_state == 1:
            # 1 = prepare species properties
            spec_prop[spec_variable].diff_min = diff_axis[diff_res]
            spec_prop[spec_variable].diff_max = diff_axis[diff_res]
            spec_prop[spec_variable].initialize()
            # 2 = create pattern
            pattern = Pattern(env_prop, spec_prop, current_n)
            # 3 = evolve pattern
            while pattern.time < before_t:
                pattern.update_pattern_euler()
            # 4 = update pattern variables
            current_n = pattern.n
            # 5 = if next is not Turing pattern, change pattern_state to 2
            if not env_prop.int_fitness.turing(d_fix, diff_axis[diff_res]):
                pattern_state = 2
        if pattern_state == 2:
            current_n = steady_n
        spec_prop[spec_variable].diff_num = 2
        for diff_inv in range(d_var[2]-1):
            if diff_res != diff_inv:
                # prepare species properties
                if diff_res < diff_inv:
                    spec_prop[spec_variable].diff_min = diff_axis[diff_res]
                    spec_prop[spec_variable].diff_max = diff_axis[diff_inv]
                    spec_prop[spec_variable].initialize()
                    inv_diff = 1
                else:
                    spec_prop[spec_variable].diff_min = diff_axis[diff_inv]
                    spec_prop[spec_variable].diff_max = diff_axis[diff_res]
                    spec_prop[spec_variable].initialize()
                    inv_diff = 0
                # create initial pattern
                in_pattern = np.zeros(3*env_prop.pos_num)
                for spec in range(2):
                    if spec == spec_variable:
                        for pos in range(env_prop.pos_num):
                            in_pattern[spec*spec_prop[0].diff_num*env_prop.pos_num+(1-inv_diff)*env_prop.pos_num+pos] = current_n[spec*env_prop.pos_num+pos]
                    else:
                        for pos in range(env_prop.pos_num):
                            in_pattern[spec*spec_prop[0].diff_num*env_prop.pos_num+pos] = current_n[spec*env_prop.pos_num+pos]
                # create the pattern with no invaders present
                pattern = Pattern(env_prop, spec_prop, in_pattern)
                # get the exponent
                inv_exponents[diff_inv, diff_res] = find_invasion_exponent(pattern, spec_variable, inv_diff, after_t)
    colors = ["#4d4d4dff", "#ffffffff", "#d4aa00ff"]
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_cmp', colors)
    im = ax.pcolormesh(diff_axis, diff_axis, inv_exponents, cmap=my_cmap, vmin=-1, vmax=1)
    # draw vertical lines for boundary of the pattern formation region
    if spec_variable == 0:
        critical_d = env_prop.int_fitness.critical_d0(d_fix)
    else:
        critical_d = env_prop.int_fitness.critical_d1(d_fix)
    ax.axvline(x=critical_d, color='black', linewidth=2, linestyle="--")
    # labels
    ax.set_xticks([d_var[0], d_var[1]])
    ax.set_yticks([d_var[0], d_var[1]])
    ax.tick_params(axis='both', which='major', labelsize=12)
    if labels:
        colors_labels = ["#214478ff", "#aa0000ff"]  # [blue, red]
        if spec_variable == 0:
            ax.set_xlabel("resident $d_A$", fontsize=12, labelpad=-8, color=colors_labels[0])
            ax.set_ylabel("invader $d_A$", fontsize=12, labelpad=-7, color=colors_labels[0])
        else:
            ax.set_xlabel("resident $d_I$", fontsize=12, labelpad=-8, color=colors_labels[1])
            ax.set_ylabel("invader $d_I$", fontsize=12, labelpad=-7, color=colors_labels[1])

# Plot how mutant perturbations evolve in time
# Measure invasion exponent
# Input: initialized pattern (1 and 2 diffusivity values for each species),
#        inv_spec (species with 2 diffusivities), inv_diff (diffusibility corresponding to invasion - 0,1)
#        before_t (time before invasion), after_t (time after invasion), seed (seed before the perturbation)
def plot_mut_pert(pattern, inv_spec, inv_diff, after_t, neutral_line, seed, ax, labels):
    # define colors
    colors = ["#d4aa00ff", "#4d4d4dff"]    # [yellow, dark grey]
    # save mutant abundance
    times = []
    mutants = []
    # introduce small proportion of invaders
    np.random.seed(seed)
    pattern.active[inv_spec * pattern.spec_prop[0].diff_num + inv_diff] = True
    for pos in range(pattern.env_prop.pos_num):
        i = inv_spec * pattern.spec_prop[0].diff_num * pattern.env_prop.pos_num + inv_diff * pattern.env_prop.pos_num + pos
        pattern.n[i] = pattern.spec_prop[inv_spec].mut_size * np.random.rand()
    m_0 = pattern.find_n_diff_tot(inv_diff, inv_spec)
    # plot neutral line
    if neutral_line:
        ax.axhline(y=m_0, color='black', linestyle=":")
    # continue evolution for another time after_t
    while pattern.time<after_t:
        times.append(pattern.time)
        mutants.append(pattern.find_n_diff_tot(inv_diff, inv_spec))
        pattern.update_pattern_euler()
    m_t = pattern.find_n_diff_tot(inv_diff, inv_spec)
    if m_t>m_0:
        ax.plot(times, mutants, color=colors[0])
    else:
        ax.plot(times, mutants, color=colors[1])
    ax.set_xlim([0, after_t])
    ax.set_yticks([])
    ax.set_xticks([0, after_t])
    ax.tick_params(axis='both', which='major', labelsize=12)
    if labels:
        ax.set_xlabel("time $t$")
        ax.set_ylabel("total number of invaders")