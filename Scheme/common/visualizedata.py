import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from common.helper import get_error_rate_grid, round_down, round_up, get_cached_error_rates, get_fusion_plot_settings, process_powerful_adv
from const.globconst import *


# Code adapted from here: https://stackoverflow.com/questions/50192121/custom-color-palette-intervals-in-seaborn-heatmap
def create_nonlin_color_dict(steps, hexcol_array):
    # Color dictionary to be returned
    cdict = {'red': (), 'green': (), 'blue': ()}
    
    # Iterate over colors
    for s, hexcol in zip(steps, hexcol_array):
        # Get RGB from hex color
        rgb = colors.hex2color(hexcol)
        
        # Populate dictionary
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
        
    return cdict
    
    
def plot_in_car_similarity(error_rates, path_helper):
    # Get grid and labels
    grid, labels = get_error_rate_grid(error_rates)
    
    # Hex color scheme, more examples here: http://colorbrewer2.org
    hc = ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704']
#     hc = ['#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636']
    
    # Thresholds
    th = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    
    # Create customized color dict
    cdict = create_nonlin_color_dict(th, hc)
    
    # Create customized colormap
    cmap = colors.LinearSegmentedColormap('zip', cdict)
    
    # Set masked color
    cmap.set_bad('black')
    
    # Define figure's size
    plt.figure(figsize=(4,4))
    
    # Copy the grid
    grid_cpy = np.copy(grid)
    
    # Replace diagonal elements with 0.0
    np.place(grid_cpy, grid_cpy == -1.0, 1.0)
    
    # Get rounded min and max values
    rmin = round_down(np.nanmin(grid_cpy), 1)
    rmax = round_up(np.nanmax(grid), 1)
    
    # Addition for ticks range
    r_add = 0.1
    
    # Handle weird case with 0.7
    if rmin == 0.7:
        r_add = 0
    
    # Ticks range
    ticks_rng = np.arange(rmin, rmax + r_add, 0.1)
    
    # Plot heatmap
    ax = sns.heatmap(grid,
            vmin=rmin, vmax=rmax,
            cmap=cmap,
            annot=True,
            mask=grid == -1.0,
            fmt='.2f',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'ticks': ticks_rng},
            annot_kws={'size':12})                             # Adjust text size of annotations
    
    # This is just a fix for matplotlib 3.1.1 
    # (https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    # Adjust font size for axes tick labels
    ax.tick_params(axis = 'both', labelsize=14)
    
    # Adjust font size for colorbar tick labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    # Rotate yticks horizontally
    plt.yticks(rotation=0)
    
    # Set axes titles
    plt.xlabel('Sensors', fontsize=14)
    plt.ylabel('Sensors', fontsize=14)
    
    # Check the scenario
    if path_helper[3] is None:
        path_helper[3] = 'full'
        
    # Check the car
    if labels == CAR1:
        car = 'car1'
    elif labels == CAR2:
        car = 'car2'
    else:
        print('plot_in_car_similarity: mixed labels "%s", should only be from CAR1 or CAR2!' % (labels,))
        return
     
    # Create filepath    
    filepath = PLOT_PATH + '/error_rates/' + path_helper[0] + '/' + path_helper[1] + '/' + path_helper[2] + '/matrix/' +  path_helper[4]
    
   # Create filepath if it does not exist
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Save figure
    plt.savefig(filepath + '/' + path_helper[3] + '-' + car + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    
    # Show figure
#     plt.show()
    
    
def plot_error_rates(filepath, action, save=[]):
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('plot_error_rates: %s is not a valid path!' % (filepath, ))
        return
    
   # Check if we are using correct data paths
    if action == 'benign':
        if not 'benign' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
    elif action == 'baseline':
        if not 'baseline' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
    elif action == 'replay':
        if not 'replay' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
    elif action == 'replay-compensation':
        if not 'replay-compensation' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
        
        # Check that we are not providing diff-park as input
        if 'diff-park' in filepath:
            print('plot_error_rates: for the diff cars experiment provide path either to "%s" or "%s" folders!' 
                  % (DIFF_NON_ADV, DIFF_ADV))
            return
        
    elif action == 'pairing-time':
        if not 'pairing-time' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return 
    else:
        print('Action "%s" does not yet have clause here!' % (action,))
        return

    # Check if we are dealing with indiv modalities or sensor fusion
    err_setup = filepath.split('/')[-1]
    
    # Set order of plotting sensors or their combinations
    if err_setup == 'indiv':
        plot_labels = ['acc_v', 'acc_h', 'gyrW', 'bar']  
        plot_labels_abbr = ['Acv', 'Ach', 'Gyr', 'Bar']
        
    elif err_setup == 'fused':
        plot_labels = ['acc_v-acc_h', 'acc_v-gyrW', 'acc_v-bar', 'acc_h-gyrW', 'acc_h-bar', 'gyrW-bar',
                      'acc_v-acc_h-gyrW', 'acc_v-acc_h-bar', 'acc_v-gyrW-bar', 'acc_h-gyrW-bar',
                      'acc_v-acc_h-gyrW-bar'] 
                      
        plot_labels_abbr = ['Acv+Ach', 'Acv+Gyr', 'Acv+Bar', 'Ach+Gyr', 'Ach+Bar', 'Gyr+Bar',
                           'Acv+Ach\n+Gyr', 'Acv+Ach\n+Bar', 'Acv+Gyr\n+Bar', 'Ach+Gyr\n+Bar',
                           'Acv+Ach+\nGyr+Bar'] 
    else:
        print('plot_error_rates: filepath: "%s" must contain either "indiv" or "fused" folder!' % (filepath))
        return
    
    # List of subscenarios
    plot_subs = ['full', 'city', 'country', 'highway', 'parking']
    
    # Load and structure cached error rates
    if action == 'replay-compensation':
        car1_plot, car2_plot = get_cached_error_rates(filepath, True)
    else:
        car1_plot, car2_plot = get_cached_error_rates(filepath)
    
    # Should you need a graph as in the paper, parking time can be suppressed like this
#     car1_plot['acc_v']['parking'] = 0
#     car2_plot['acc_v']['parking'] = 0
    
    # Check if we need to save figures or just display them
    if save:
        # Check if save is a list
        if not isinstance(save, list):
            print('plot_error_rates: "save" must be provided as list of strings!')
            return
        
        # Sort save list
        save.sort()
        
        # Check save is valid
        if save != ['1'] and save != ['2'] and save != ['1', '2']:
            print('plot_error_rates: "save" can only be ["1"], ["2"], or ["1", "2"]')
            return
        
        # Iterate over cars
        for s in save:
            # Define figure's size
            fig = plt.figure(figsize=(11, 8))
            
            # Index to track iterations
            idx = 0

            # Bar width
            bar_width = 0.09

            # X spacing
            x_spacing = 0.04
            
            # Iterate over subscenarios
            for sub in plot_subs:
                
                # Adjust X-axis
                if idx == 0:
                    # Positions on X-axis
                    x_axis = np.arange(len(plot_labels))
                else:
                    x_axis = [x + bar_width + x_spacing for x in x_axis]
                    
                # Store Y-asix values and Y-axis errors
                y_axis = []
                y_axis_err = []
                
                # Iterate over modalities
                for label in plot_labels:
                    # Check which car to plot (defined by save list)
                    if s == '1':
                        # Compute average and std for a specific sensor type in a specific subscenario (CAR1)
                        y_axis.append(np.mean(car1_plot[label][sub]))
                        y_axis_err.append(np.std(car1_plot[label][sub]))
                        
                    elif s == '2':
                        # Compute average and std for a specific sensor type in a specific subscenario (CAR2)
                        y_axis.append(np.mean(car2_plot[label][sub]))
                        y_axis_err.append(np.std(car2_plot[label][sub]))
                
                # Plot stuff 
                plt.bar(x_axis, y_axis, yerr=y_axis_err, width=bar_width, edgecolor='black', linewidth=0.75,
                        label=sub.capitalize(), zorder=3, capsize=6)
                
                # Increment idx
                idx += 1
            
            # Add grid to a plot
            plt.grid(True, axis='y', zorder=0)
            
            # Setup ticks fontsize
            plt.tick_params(axis='both', labelsize=20)
            
            # Y-axis range depends on the scenario
            if action == 'benign':
                y_range = np.arange(0, 1.1, step=0.1)
            elif action == 'baseline':
                if 'silent' in filepath:
                    y_range = np.arange(0, 0.065, step=0.01)
                elif 'moving' in filepath:
                    y_range = np.arange(0, 0.065, step=0.01)
            elif action == 'replay':
                y_range = np.arange(0, 0.26, step=0.05)
            elif action == 'replay-compensation':
                y_range = np.arange(0, 0.51, step=0.1)
            elif action == 'pairing-time':
                if 'fpake1' in filepath:
#                     y_range = np.arange(0, 176, 25)
                    y_range = np.arange(0, 126, 25)
                elif 'fpake' in filepath:
                    y_range = np.arange(0, 101, 20)
                elif 'fcom' in filepath:
#                     y_range = np.arange(90, 900, 100)
#                     y_range = np.arange(0, 301, 25)
                    y_range = np.arange(0, 226, 25)
            
            # Add xticks in the middle of the group bars
            if len(plot_subs) == 4:
                plt.xticks([r + (bar_width * 2 + x_spacing / 2) for r in range(len(plot_labels))], plot_labels_abbr)
                plt.yticks(y_range)  
            elif len(plot_subs) == 5:
                plt.xticks([r + (bar_width + x_spacing) * 2 for r in range(len(plot_labels))], plot_labels_abbr)
                plt.yticks(y_range) 
                
            # X-axis name
            plt.xlabel('Sensor Modality', fontsize=24)
            
            # Y-axis name
            if action == 'pairing-time':
                plt.ylabel('Pairing Time (sec)', fontsize=24)
            elif action == 'benign':
                plt.ylabel('True Acceptance Rate (TAR)', fontsize=24)
            else:
                plt.ylabel('False Acceptance Rate (FAR)', fontsize=24)
            
            # Legend
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=len(plot_subs), mode="expand", borderaxespad=0., 
                      fontsize=19)
            
            # Plot filepath components
            exp = ''
            bl_mode = ''
            rpl_exp = ''
            
            # Construct plot title and saving filepath
            if action == 'benign' or action == 'baseline' or action == 'pairing-time':
                if 'sim' in filepath:
                    exp = 'sim'
                elif 'diff' in filepath:
                    exp = 'diff'
                else:
                    print('plot_error_rates: either "sim" or "diff" must be present in the cache path: %s!' % (filepath,))
                    return
                
                # Special case for baseline
                if action == 'baseline':
                    if 'silent' in filepath:
                        bl_mode = 'silent'
                    elif 'moving' in filepath:
                        bl_mode = 'moving'
                        
                # Special case for pairing-time
                if action == 'pairing-time':
                    if 'fpake' in filepath:
                        bl_mode = 'fpake'
                    elif 'fcom' in filepath:
                        bl_mode = 'fcom'
            
            elif action == 'replay' or action == 'replay-compensation':
                # Get experiment name 
                regex = r'(?:/|\\)far(?:/|\\)(.*)(?:/|\\)' + re.escape(err_setup)
                match = re.search(regex, filepath)

                # If there is no match - exit
                if not match:
                    print('plot_error_rates: no match for the file name %s using regex %s!' % (filepath, regex))
                    return
                
                exp = match.group(1)
                
                # Case for the replay-compensation
                if action == 'replay-compensation':
                    exp = exp.split('/')[0]
                
                # Abbreviation
                if exp == 'sim-non-adv':
                    rpl_exp = 'sna'
                elif exp == 'sim-adv':
                    rpl_exp = 'sadv'
                elif exp == 'diff-non-adv':
                    rpl_exp = 'dna'
                elif exp == 'diff-adv':
                    rpl_exp = 'dadv'
            
            # Plot title
            if bl_mode:
                plot_title = action.capitalize() + ' (' + bl_mode + ')'  + ': similar cars-' + s
            else:
                plot_title = action.capitalize() + ': similar cars-' + s
                
            # Plot saving filepath  
            if bl_mode:
                plot_filepath = PLOT_PATH + '/error_rates/' + action + '/' + bl_mode + '/' + exp + '/' + err_setup
            else:
                plot_filepath = PLOT_PATH + '/error_rates/' + action + '/' + exp + '/' + err_setup
            
            # Construct filename
            if bl_mode:
                # Baseline case
                plot_filename = action + '-' + err_setup + '-' + bl_mode + '-' + exp + '-' + 'car' + s
            else:
                # Replay case
                if rpl_exp:
                    plot_filename = action + '-' + err_setup + '-' + rpl_exp + '-' + 'car' + s
                else:
                    # Benign case
                    plot_filename = action + '-' + err_setup + '-' + exp + '-' + 'car' + s
            
            # Create fp_path if it does not exist
            if not os.path.exists(plot_filepath):
                os.makedirs(plot_filepath)
            
            # Save plot 
            plt.savefig(plot_filepath + '/' + plot_filename + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight')
            
            # Show plot
            plt.show()
    else:
        print()

        # Index to track iterations
        idx = 0

        # Bar width
        bar_width = 0.09

        # X spacing
        x_spacing = 0.04

        # Figure to be plotted
        fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True)

        # Plot filepath components
        exp = ''
        bl_mode = ''
        rpl_exp = ''

        # Construct plot title and saving filepath
        if action == 'benign' or action == 'baseline' or action == 'pairing-time':
            if 'sim' in filepath:
                exp = 'sim'
            elif 'diff' in filepath:
                exp = 'diff'
            else:
                print('plot_error_rates: either "sim" or "diff" must be present in the cache path: %s!' % (filepath,))
                return

            # Special case for baseline
            if action == 'baseline':
                if 'silent' in filepath:
                    bl_mode = 'silent'
                elif 'moving' in filepath:
                    bl_mode = 'moving'
            
            # Special case for pairing-time
            if action == 'pairing-time':
                if 'fpake' in filepath:
                    bl_mode = 'fpake'
                elif 'fcom' in filepath:
                    bl_mode = 'fcom'
        
        elif action == 'replay' or action == 'replay-compensation':
            # Get experiment name 
            regex = r'(?:/|\\)far(?:/|\\)(.*)(?:/|\\)' + re.escape(err_setup)
            match = re.search(regex, filepath)

            # If there is no match - exit
            if not match:
                print('plot_error_rates: no match for the file name %s using regex %s!' % (filepath, regex))
                return

            exp = match.group(1)
            
            # Case for the replay-compensation
            if action == 'replay-compensation':
                exp = exp.split('/')[0]
            
            # Abbreviation
            if exp == 'sim-non-adv':
                rpl_exp = 'sna'
            elif exp == 'sim-adv':
                rpl_exp = 'sadv'
            elif exp == 'diff-non-adv':
                rpl_exp = 'dna'
            elif exp == 'diff-adv':
                rpl_exp = 'dadv'
        
        # Plot title
        if bl_mode:
            plot_title = action.capitalize() + ' (' + bl_mode + ')'  + ': ' + exp + ' cars'
        else:
            plot_title = action.capitalize() + ': ' + exp + ' cars'
        
        fig.suptitle(plot_title, y=1.03, fontsize=26, fontweight='bold')

        # Iterate over subscenarios
        for sub in plot_subs:
            # Adjust X axis
            if idx == 0:
                # Positions on X-axis
                x_axis = np.arange(len(plot_labels))
            else:
                x_axis = [x + bar_width + x_spacing for x in x_axis]

            # Store Y-asix values and Y-axis errors
            y_axis1 = []
            y_axis_err1 = []
            y_axis2 = []
            y_axis_err2 = []

            # Iterate over modalities
            for label in plot_labels:
                # Compute average and std for a specific sensor type in a specific subscenario (CAR1)
                y_axis1.append(np.mean(car1_plot[label][sub]))
                y_axis_err1.append(np.std(car1_plot[label][sub]))

                # Compute average and std for a specific sensor type in a specific subscenario (CAR1)
                y_axis2.append(np.mean(car2_plot[label][sub]))
                y_axis_err2.append(np.std(car2_plot[label][sub]))

                # Display some stat
                print('CAR1,%s,%s,%.3f,%.5f' % (sub, label, np.mean(car1_plot[label][sub]), np.std(car1_plot[label][sub])))
                print('CAR2,%s,%s,%.3f,%.5f' % (sub, label, np.mean(car2_plot[label][sub]), np.std(car2_plot[label][sub])))
            print()

            # Plot stuff 
            axs[0].bar(x_axis, y_axis1, yerr=y_axis_err1, width=bar_width, label=sub.capitalize(), capsize=10, zorder=3)
            axs[1].bar(x_axis, y_axis2, yerr=y_axis_err2, width=bar_width, label=sub.capitalize(), capsize=10, zorder=3)

            # Increment idx
            idx += 1

        # Add names to subplots
        axs[0].set_title('CAR 1', fontsize=22)
        axs[1].set_title('CAR 2', fontsize=22)

        # Add grid to subplots
        axs[0].grid(True, axis='y', zorder=0)
        axs[1].grid(True, axis='y', zorder=0)

        # Y-axis range depends on the scenario
        if action == 'benign':
            y_range = np.arange(0, 1.1, step=0.1)
        elif action == 'baseline':
            y_range = np.arange(0, 0.09, step=0.02)
        elif action == 'replay':
            y_range = np.arange(0, 0.26, step=0.05)
        elif action == 'replay-compensation':
            y_range = np.arange(0, 0.41, step=0.05)
        elif action == 'pairing-time':
            if 'fpake' in filepath:
                y_range = np.arange(0, 176, 25)
            elif 'fcom' in filepath:
                y_range = np.arange(90, 900, 100)
#                 y_range = np.arange(0, 301, 25)
            
        # Add xticks in the middle of the group bars
        if len(plot_subs) == 4:
            plt.setp(axs, xticks=[r + (bar_width * 2 + x_spacing / 2) for r in range(len(plot_labels))], xticklabels=plot_labels_abbr, 
                    yticks=y_range)  
        elif len(plot_subs) == 5:
            plt.setp(axs, xticks=[r + (bar_width + x_spacing) * 2 for r in range(len(plot_labels))], xticklabels=plot_labels_abbr, 
                    yticks=y_range)   
        
        # Adjust label sizes
        axs[0].xaxis.set_tick_params(labelsize=20)
        axs[0].yaxis.set_tick_params(labelsize=20)
        axs[1].xaxis.set_tick_params(labelsize=20)
        axs[1].yaxis.set_tick_params(labelsize=20)

        # Add legends
        axs[0].legend(loc='upper right', fontsize=20, ncol=len(plot_subs))
        axs[1].legend(loc='upper right', fontsize=20, ncol=len(plot_subs))

        # Add tight layout for figure
        fig.tight_layout()

       # Plot saving filepath  
        if bl_mode:
            plot_filepath = '/home/seemoo/plots-png/error_rates' + '/' + action + '/' + bl_mode +'/' + exp +'/' + err_setup
        else:
            plot_filepath = '/home/seemoo/plots-png/error_rates' + '/' + action + '/' + exp +'/' + err_setup
        
        # Construct filename
        if bl_mode:
            # Baseline case
            plot_filename = action + '-' + err_setup + '-' + bl_mode + '-' + exp + '-' + 'car12'
        else:
            # Replay case
            if rpl_exp:
                plot_filename = action + '-' + err_setup + '-' + rpl_exp + '-' + 'car12'
            else:
                # Benign case
                plot_filename = action + '-' + err_setup + '-' + exp + '-' + 'car12'
        
        # Create fp_path if it does not exist
        if not os.path.exists(plot_filepath):
            os.makedirs(plot_filepath)

        # Save plot 
        plt.savefig(plot_filepath + '/' + plot_filename + '.png', format='png', dpi=300)

        # Show plot
        plt.show()
    
    return


def plot_fusion_effect(filepath, action, plot_setup={}):
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('plot_fusion_effect: %s is not a valid path!' % (filepath, ))
        return
    
    # Check if we are using correct data paths
    if action == 'benign':
        if not 'benign' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
    elif action == 'baseline':
        if not 'baseline' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
    elif action == 'replay':
        if not 'replay' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
    elif action == 'replay-compensation':
        if not 'replay-compensation' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
        
        # Check that we are not providing diff-park as input
        if 'diff-park' in filepath:
            print('plot_fusion_effect: for the diff cars experiment provide path either to "%s" or "%s" folders!' 
                  % (DIFF_NON_ADV, DIFF_ADV))
            return  
    
    elif action == 'powerful':
        if not 'powerful' in filepath:
            print('Warning: double check if filepath points to "%s" cache files!' % (action,))
            return
    else:
        print('Action "%s" does not yet have clause here!' % (action,))
        return

    # Check if we are dealing with sensor fusion
    err_setup = filepath.split('/')[-1]
    
    if err_setup != 'fused' and action != 'powerful':
        print('plot_fusion_effect: filepath "%s" must point to the sensor fusion folder!' % (filepath))
        return
    else:
        labels_fused = ['acc_v-acc_h', 'acc_v-gyrW', 'acc_v-bar', 'acc_h-gyrW', 'acc_h-bar', 'gyrW-bar',
                      'acc_v-acc_h-gyrW', 'acc_v-acc_h-bar', 'acc_v-gyrW-bar', 'acc_h-gyrW-bar',
                      'acc_v-acc_h-gyrW-bar']  
        
        labels_indiv = ['acc_v', 'acc_h', 'gyrW', 'bar']  
        
        plot_labels_abbr = ['Acv (V)\nAch (H)\nGyr (G)\nBar (B)', 'V+H\nH+V\nG+V\nB+V', 'V+G\nH+G\nG+H\nB+H', 
                            'V+B\nH+B\nG+B\nB+G', 'V+H+G\nH+V+G\nG+V+H\nB+V+H', 'V+H+B\nH+V+B\nG+V+B\nB+V+G', 
                            'V+G+B\nH+G+B\nG+H+B\nB+H+G', 'All'] 
    
    # Scenarios we are dealing with
#     plot_subs = ['full', 'city', 'country', 'highway']
    plot_subs = ['full', 'city', 'country', 'highway', 'parking']
    
    # Check if plot_setup is a valid dict
    if not isinstance(plot_setup, dict) or len(plot_setup) < 1:
        print('plot_fusion_effect: "plot_setup" must be a non-empty dictionary!')
        return
    
    # Check if plot_setup has valid keys and values
    if not set(list(plot_setup.keys())).issubset(labels_indiv):
        print('plot_fusion_effect: invalid keys in "plot_setup", only %s are allowed!' % (labels_indiv))
        return
        
    if not set(list(plot_setup.values())).issubset(plot_subs):
        print('plot_fusion_effect: invalid values in "plot_setup", only %s are allowed!' % (plot_subs))
        return
    
    # Load and structure cached error rates
    if action == 'replay-compensation':
        car1_plot_fused, car2_plot_fused = get_cached_error_rates(filepath, True)
        car1_plot_indiv, car2_plot_indiv = get_cached_error_rates(filepath.replace(err_setup, 'indiv'), True)
    
    elif action == 'powerful':
        if 'car1-2' in filepath:
            car1_plot_fused, _ = get_cached_error_rates(filepath, powerful=True)
            _, car2_plot_fused = get_cached_error_rates(filepath.replace('car1-2', 'car2-1'), powerful=True)
            
        elif 'car2-1':
            _, car2_plot_fused = get_cached_error_rates(filepath, powerful=True)
            car1_plot_fused, _ = get_cached_error_rates(filepath.replace('car2-1', 'car1-2'), powerful=True)
            
        else:
            print('plot_fusion_effect: "powerful" filepath must point to either "car1-2" or "car2-1" folders!')
            return
        
        # Do extra processing
        car1_plot_indiv, car1_plot_fused = process_powerful_adv(car1_plot_fused, plot_setup)
        car2_plot_indiv, car2_plot_fused = process_powerful_adv(car2_plot_fused, plot_setup)
        
    else:
        car1_plot_fused, car2_plot_fused = get_cached_error_rates(filepath)
        car1_plot_indiv, car2_plot_indiv = get_cached_error_rates(filepath.replace(err_setup, 'indiv'))
    
    print()
    
    # Plot settings: default color scheme: https://matplotlib.org/3.1.1/users/dflt_style_changes.html
    plot_settings = get_fusion_plot_settings(plot_setup)
    
    # Construct filepath for saving plots
    plot_setup_str = ''
    
    for k,v in plot_setup.items():
        plot_setup_str += k + '-' + v + '+'
    
    # Plot filepath components
    exp = ''
    bl_mode = ''
    rpl_exp = ''
    
    # Construct plot title and saving filepath
    if action == 'benign' or action == 'baseline':
        if 'sim' in filepath:
            exp = 'sim'
        elif 'diff' in filepath:
            exp = 'diff'
        else:
            print('plot_fusion_effect: either "sim" or "diff" must be present in the cache path: %s!' % (filepath,))
            return

        # Special case for baseline
        if action == 'baseline':
            if 'silent' in filepath:
                bl_mode = 'silent'
            elif 'moving' in filepath:
                bl_mode = 'moving'
    
    elif action == 'replay' or action == 'replay-compensation' or action == 'powerful':
        # Get experiment name 
        regex = r'(?:/|\\)far(?:/|\\)(.*)(?:/|\\)' + re.escape(err_setup)
        match = re.search(regex, filepath)

        # If there is no match - exit
        if not match:
            print('plot_fusion_effect: no match for the file name %s using regex %s!' % (filepath, regex))
            return

        exp = match.group(1)
        
        # Case for the replay-compensation
        if action == 'replay-compensation':
            exp = exp.split('/')[0]
        
        # Abbreviation
        if exp == 'sim-non-adv':
            rpl_exp = 'sna'
        elif exp == 'sim-adv':
            rpl_exp = 'sadv'
        elif exp == 'diff-non-adv':
            rpl_exp = 'dna'
        elif exp == 'diff-adv':
            rpl_exp = 'dadv'
    
    # Plot saving filepath  
    if bl_mode:
        plot_filepath = PLOT_PATH + '/error_rates/' + action + '/' + bl_mode +'/' + exp +'/' + err_setup + '/' + plot_setup_str[:-1]
    else:
        if action == 'powerful':
            plot_filepath = PLOT_PATH + '/error_rates/' + action + '/' + exp + '/' + plot_setup_str[:-1]
        else:
            plot_filepath = PLOT_PATH + '/error_rates/' + action + '/' + exp +'/' + err_setup + '/' + plot_setup_str[:-1]

    # Construct filename
    if bl_mode:
        # Baseline case
        plot_filename = action + '-' + err_setup + '-' + bl_mode + '-' + exp + '-' + 'car'
    else:
        # Replay and powerful cases
        if rpl_exp:
            if action == 'powerful':
                plot_filename = action + '-' + rpl_exp + '-' + err_setup
            else:
                plot_filename = action + '-' + err_setup + '-' + rpl_exp + '-' + 'car'
        else:
            # Benign case
            plot_filename = action + '-' + err_setup + '-' + exp + '-' + 'car'
    
    # Create fp_path if it does not exist
    if not os.path.exists(plot_filepath):
        os.makedirs(plot_filepath)
    
    # We want to display two cars
    for i in range(0, 2):
        # Define figure's size
        fig = plt.figure(figsize=(11, 8))
        
        # Index to adjust plot settings
        idx = 0
        
        # Iterate over individual modalities
        for ilab in labels_indiv:
            # Error rates
            error_rates = []
            
            # Append indiv modality error rate (car 1, car 2)
            if i == 0:
                error_rates.append(np.mean(np.array(car1_plot_indiv[ilab][plot_setup[ilab]])))
            else:
                error_rates.append(np.mean(np.array(car2_plot_indiv[ilab][plot_setup[ilab]])))
            
            # Iterate over fused modalities
            for flab in labels_fused:
                # Check if individ modalities is contined within the fused combination
                if ilab in flab:
                    # Append fused modalities (car 1, car 2)
                    if i == 0:
                        if flab in car1_plot_fused:
                            error_rates.append(np.mean(np.array(car1_plot_fused[flab][plot_setup[ilab]])))
                        else:
                            print('plot_fusion_effect: key "%s" must be contained in car1_plot_fused dictionary!')
                            return
                    else:
                        if flab in car2_plot_fused:
                            error_rates.append(np.mean(np.array(car2_plot_fused[flab][plot_setup[ilab]])))
                        else:
                            print('plot_fusion_effect: key "%s" must be contained in car2_plot_fused dictionary!')
                            return
            
            # Plot label
            plabel = plot_setup[ilab].capitalize()
            
            # Adjust markers
            if plot_settings[idx][1] == 'x':
                # Adjust marker size
                mew = 7
                ms = 14
            else:
                # Default marker size
                mew = 6
                ms = 12
            
            # Plot error rates
            if isinstance(plot_settings[idx][0], str):
                plt.plot(np.arange(0.125, 1.1, 0.125), error_rates, linestyle=plot_settings[idx][0], marker=plot_settings[idx][1], 
                          color=plot_settings[idx][2], label=plabel, linewidth=6, mew=mew, ms=ms) 
                
            elif isinstance(plot_settings[idx][0], list):
                plt.plot(np.arange(0.125, 1.1, 0.125), error_rates, dashes=plot_settings[idx][0], marker=plot_settings[idx][1], 
                          color=plot_settings[idx][2], label=plabel, linewidth=6, mew=mew, ms=ms) 
                
            # Increment idx
            idx += 1
        
        # Adjust axes ticks
        plt.xticks(np.arange(0.125, 1.1, 0.125), plot_labels_abbr)
        if action == 'benign':
            plt.yticks(np.arange(0.6, 1.05, 0.05))
        elif action == 'baseline':
            if bl_mode == 'silent':
                plt.yticks(np.arange(0, 0.06, step=0.01))
            elif bl_mode == 'moving':
                plt.yticks(np.arange(0, 0.065, step=0.01))
        elif action == 'replay':
            plt.yticks(np.arange(0, 0.26, step=0.05))
        elif action == 'replay-compensation':
            plt.yticks(np.arange(0, 0.41, step=0.05))
        elif action == 'powerful':
            plt.yticks(np.arange(0, 0.91, step=0.1))
        
        # Set up ticks fontsize
        plt.tick_params(axis='both', labelsize=20)
        
        # X-axis title
        plt.xlabel('Sensor Modalities: Individual and Fused', fontsize=24)
        
        # Y-axis title
        if action == 'benign':
            plt.ylabel('True Acceptance Rate (TAR)', fontsize=24)
        else:
            plt.ylabel('False Acceptance Rate (FAR)', fontsize=24)
        
        # Legend
        legend1 = plt.legend(loc='best', fontsize=19)
        
        plt.legend(['', '', '', ''], loc=2, frameon=False, fontsize=15, bbox_to_anchor=(-0.09, 0.01))
        plt.gca().add_artist(legend1)
        
        # Add grid
        plt.grid(True, axis='y')
        
        # Add tight layout for figure
        fig.tight_layout()
        
        # Save plot 
        if i == 0: 
            if action == 'powerful':
                plt.savefig(plot_filepath + '/' + plot_filename + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight')
            else:
                plt.savefig(plot_filepath + '/' + plot_filename + '1.pdf', format='pdf', dpi=1000, bbox_inches = 'tight')
        else:
            if action == 'powerful':
                if 'car1-2' in plot_filename:
                    plot_filename =  plot_filename.replace('car1-2', 'car2-1')
                    
                elif 'car2-1' in plot_filename:
                    plot_filename =  plot_filename.replace('car2-1', 'car1-2')
                
                plt.savefig(plot_filepath + '/' + plot_filename + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight')
            
            else:
                plt.savefig(plot_filepath + '/' + plot_filename + '2.pdf', format='pdf', dpi=1000, bbox_inches = 'tight')
        
        # Show plot
        plt.show()
