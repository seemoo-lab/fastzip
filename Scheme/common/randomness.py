import sys
import random
import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.special import comb
from scipy.special import binom
from cycler import cycler
from scipy.special import comb

# The code is adapted from here:
# https://github.com/seemoo-lab/ubicomp19_zero_interaction_security/blob/master/Visualization/SchuermannSigg.ipynb


def random_walk(key, sum_distribution, transition_count, transition_probs, bitlength=24):
    ''' takes string of '0' and '1' and turns them into random walks in a galton board. Effectively computes
    the cumulative sums distribution for every prefix of the input string. '''

    transition_count[0] += 1
    ret = [0]
    for i, b in enumerate(key[:bitlength]):
        val = -1
        if b == '1':
            val = 1
            transition_probs[i] += 1
        ret.append(ret[-1] + val)

    sum_distribution.append(ret[-1])
    return (sum_distribution, transition_count, transition_probs)


def markov_transitions(transition_probs, transition_count, bits=24, save_to=None):
    ''' transition probabilities from every nth to (n+1)th bit. '''
    norm_transition_probs = [transition_probs[x] / transition_count[0]
                             for x in range(0, bits)]
    plt.clf()
    fig = plt.figure(figsize=(7, 4))
    plt.xlabel('N-th bit', fontsize=20)
    plt.ylabel('Probability of 1-bit', fontsize=20)
    plt.ylim([0, 1])
    plt.xlim([0, bits])
    plt.rcParams.update({'font.size': 24})
    plt.plot(norm_transition_probs, color='b', linewidth=4)
    plt.plot([0.0, bits], [0.5, 0.5], 'k:', linewidth=4)
    if save_to is not None:
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(save_to, format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()
    print("Markov:", np.median(norm_transition_probs))

    
def binomial_plot(n=256):
    ''' Theoretical binomial distribution that is plotted as red line into figure.'''
    x = np.arange(n)
    # y = list(map(lambda xi: comb(n,xi)/2**n, x))
    y = list(binom(n,x)*0.5**x*(0.5)**(n-x))
    x = list(map(lambda r: r-n/2, x))
    plt.plot(x,y,color='r',linewidth=4)


def distribution(sum_distribution, bits=24, dist_xlim=None, save_to=None):
    ''' Plots the cumulative sums distribution and saves figure. '''
    plt.clf()
    fig = plt.figure(figsize=(7, 4))
    if dist_xlim is not None:
        plt.xlim(dist_xlim)
    else:
        plt.xlim([-bits, bits])
    count, bins, ignored = plt.hist(sum_distribution, color='#007a9b', range=(-bits,bits), density=True, rwidth=0.5, bins=bits)
    binomial_plot(bits*2)
    # Get the current axes
    ax = plt.gca()
    # Get limits
    start, end = ax.get_xlim()

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    if save_to is not None:
        plt.savefig(save_to, format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()
    print("Median of distribution:", np.median(sum_distribution))
    

def plot_rand_walk(keys, bits=24, dist_xlim=None, save_distribution_to=None, save_markov_to=None):
    ''' turns string consisting of '0' and '1' into random walks. While the walks are currently not plotted, the distribution
    of the cumulative sums along with the markov transitions are computed and plotted using them.'''
    sum_distribution = []
    transition_count = [0]
    transition_probs = {x: 0 for x in range(0, bits)}

    matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                    ['#e78a33', '#eda766', '#8d4959', '#aa7782', '#bdcd61', '#cdda89',
                                                     '#8a9c33', '#a7b566'])
    plt.rcParams.update({'font.size': 18})

    plt.ylabel('Sum')
    plt.xlabel('Keylength')

    plt.ylim([-bits, bits])
    for key in keys:
        sum_distribution, transition_count, transition_probs = random_walk(key, sum_distribution, transition_count, transition_probs, bits)
    plt.tight_layout()
    plt.rcParams.update({'font.size': 18})
    distribution(sum_distribution, bits, dist_xlim=dist_xlim, save_to=save_distribution_to)
    markov_transitions(transition_probs, transition_count, bits, save_to=save_markov_to)
