########################
#     Libraries        #
########################
import os
import numpy as np
import matplotlib.pyplot as plt
import methods as mtd
import matplotlib.colors as clr
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib import collections  as mc
import pandas as pd

########################
#      Input           #
########################
nu                    = 0.10
S                     = 1e6
T                     = 220
K_range               = range(0,200,1)#range(86,87,1)
L                     = 200
moving_average_period = 220
xaxisfontsize         = 20
yaxisfontsize         = 18
ticklabelsize         = 16
legendfontsize        = 20
figuretagsize         = 20
confidence_level      = 1.96
########################
#     Analysis type    #
########################
general_analysis      = True
specific_analysis     = False
target_experiments    = [93,99,50,48,57,126,28,151]
########################
#      Folder          #
########################
common_path  = "D:\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\risk analysis\\" #"\\extensions (1e9)\\"
figure_path  = common_path + "figures\\time series analysis\\"
os.makedirs(figure_path, exist_ok=True)

if general_analysis:
            cCMP_folder = common_path  + "{:.2f}".format(nu) + "\\cCMP\\"
            CMP_folder  = common_path  + "{:.2f}".format(nu) + "\\CMP\\"
            MF_folder   = common_path + "{:.2f}".format(nu) + "\\MF\\"
            for sim_id in K_range:

                    cCMP_nodes_per_step    = np.load(cCMP_folder + 'experiment_' + str(sim_id) + '\\active_nodes_cCMP_' + str(sim_id) + '.npy')
                    cCMP_nodes_per_step_ma = pd.rolling_mean(cCMP_nodes_per_step, moving_average_period)
                    CMP_nodes_per_step     = np.load(CMP_folder + 'experiment_' + str(sim_id) + '\\active_nodes_CMP_' + str(sim_id) + '.npy')
                    #CMP_nodes_per_step_ma  = []
                    CMP_nodes_per_step_ma  = pd.rolling_mean(CMP_nodes_per_step, moving_average_period)
                    #for i in range(moving_average_period,np.int(S),100):
                        #avg = np.average(CMP_nodes_per_step[(i-moving_average_period):i])
                        #CMP_nodes_per_step_ma.append(avg)
                        #print(i)
                    timeline_cCMP          = np.arange(0, len(cCMP_nodes_per_step),1)
                    timeline_CMP            = np.arange(0, len(CMP_nodes_per_step), 1)
                    n_structures_cCMP       = len(np.load(cCMP_folder + 'experiment_' + str(sim_id) + '\\cs_length_cCMP_' + str(sim_id) + '.npy'))
                    n_structures_CMP        = len(np.load(CMP_folder + 'experiment_' + str(sim_id) + '\\cs_length_CMP_' + str(sim_id) + '.npy'))
                    time_in_AF_cCMP        = np.load(cCMP_folder + 'experiment_' + str(sim_id) + '\\af_risk_cCMP_' + str(sim_id) + '.npy').item()
                    time_in_AF_CMP          = np.load(CMP_folder + 'experiment_' + str(sim_id) + '\\af_risk_CMP_' + str(sim_id) + '.npy').item()
                    time_in_AF_MF          = np.load(MF_folder + 'experiment_' + str(sim_id) + '\\af_risk_MF_' + str(sim_id) + '.npy').item()
                    MF_structures_per_step  = np.load(MF_folder + 'experiment_' + str(sim_id) + '\\active_particles_MF' + str(sim_id) + '.npy')
                    MF_structures_per_step_ma = pd.rolling_mean(MF_structures_per_step, moving_average_period)

                    '''
                    fig, ax                = plt.subplots(figsize=(6, 6), dpi=200)
                    ax.plot(timeline_cCMP, cCMP_nodes_per_step, linewidth=0.05, color='k', label=r'$a(t)$')
                    ax.plot(timeline_cCMP, cCMP_nodes_per_step_ma, linewidth=1.0, color='red', label=r'$\langle a(t) \rangle$')
                    ax.set_xlabel('$t$', fontsize=xaxisfontsize)
                    ax.set_ylabel(r'$a(t)$', fontsize=yaxisfontsize)
                    ax.set_xlim([0, S])
                    ax.set_ylim([0, 5 * L + 1])
                    ax.set_yticks(range(0, 5 * L + 1, int(0.5 * L)))
                    ax.set_yticklabels(range(0, 5 * L + 1, int(0.5 * L)))
                    plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
                    plt.tick_params(which='major', length=6)
                    plt.tick_params(which='minor', length=3)
                    ax.axhline(T, linestyle='dashed', color='blue', linewidth=2.0)
                    ax.yaxis.set_label_coords(-0.18, 0.5)
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    plt.rc('font', **{'size': str(ticklabelsize)})
                    plt.subplots_adjust(left=0.20)
                    figure_path_current = figure_path + "{:.2f}".format(nu) + "\\cCMP\\"
                    os.makedirs(figure_path_current, exist_ok=True)
                    plt.savefig(figure_path_current + 'time_series_' + str(sim_id) + '.png')
                    plt.close()
                    '''
                    
                    
                    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                    #ax.plot(timeline_CMP[::100], CMP_nodes_per_step[::100], linewidth=0.05, color='k', label=r'$a(t)$')
                    ax.plot(timeline_CMP, CMP_nodes_per_step, linewidth=0.05, color='k', label=r'$a(t)$')
                    #print(np.where(CMP_nodes_per_step <= 220)[0])
                    #ax.plot(timeline_CMP[220::100], CMP_nodes_per_step_ma, linewidth=1.0, color='red',label=r'$\langle a(t) \rangle$')
                    ax.plot(timeline_CMP, CMP_nodes_per_step_ma, linewidth=1.0, color='red',label=r'$\langle a(t) \rangle$')
                    ax.axhline(T, linestyle='dashed', color='blue', linewidth=2.0)
                    ax.set_xlabel('$t$', fontsize=xaxisfontsize)
                    ax.set_ylabel(r'$a(t)$', fontsize=yaxisfontsize)
                    ax.set_xlim([0, S])
                    ax.set_ylim([0, 5 * L + 1])
                    ax.set_yticks(range(0, 5 * L + 1, int(0.5 * L)))
                    ax.set_yticklabels(range(0, 5 * L + 1, int(0.5 * L)))
                    plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
                    plt.tick_params(which='major', length=6)
                    plt.tick_params(which='minor', length=3)
                    ax.axhline(T, linestyle='dashed', color='blue', linewidth=2.0)
                    ax.yaxis.set_label_coords(-0.18, 0.5)
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    plt.rc('font', **{'size': str(ticklabelsize)})
                    plt.subplots_adjust(left=0.20)
                    figure_path_current = figure_path + "{:.2f}".format(nu) + "\\CMP\\"
                    os.makedirs(figure_path_current, exist_ok=True)
                    plt.savefig(figure_path_current + 'time_series_' + str(sim_id) + '.png')
                    plt.close()

                    '''
                    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                    ax.plot(timeline_CMP, MF_structures_per_step, linewidth=0.05, color='k', label=r'$N(t)$')
                    ax.plot(timeline_CMP, MF_structures_per_step_ma, linewidth=1.0, color='red',label=r'$\langle N(t) \rangle$')
                    ax.axhline(1, linestyle='dashed', color='blue', linewidth=2.0)
                    ax.axhline(n_structures_cCMP, linestyle='dashed', color='blue', linewidth=2.0)
                    ax.set_xlabel('$t$', fontsize=xaxisfontsize)
                    ax.set_ylabel(r'$N(t)$', fontsize=yaxisfontsize)
                    ax.set_xlim([0, S])
                    ax.set_ylim([0, n_structures_cCMP + 1])
                    ax.set_yticks(range(0, n_structures_cCMP + 1, 2))
                    ax.set_yticklabels(range(0, n_structures_cCMP + 1, 2))
                    
                    plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
                    plt.tick_params(which='major', length=6)
                    plt.tick_params(which='minor', length=3)
                    ax.yaxis.set_label_coords(-0.18, 0.5)
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    plt.rc('font', **{'size': str(ticklabelsize)})
                    plt.subplots_adjust(left=0.20)
                    figure_path_current = figure_path + "{:.2f}".format(nu) + "\\MF\\"
                    os.makedirs(figure_path_current, exist_ok=True)
                    plt.savefig(figure_path_current + 'time_series_' + str(sim_id) + '.png')
                    plt.close()
                    '''


                    print('Figure produced. Simulation id: ' + str(sim_id))
                    #print('Simple structures (cCMP, CMP): (' + str(n_structures_cCMP) + ',' + str(n_structures_CMP) +')')
                    #print('Time in AF (cCMP, CMP, MF): (' + str(time_in_AF_cCMP) + ',' + str(time_in_AF_CMP) + ')')# ',' + str(time_in_AF_MF)+ ')')


if specific_analysis:

            CMP_folder = common_path + "{:.2f}".format(nu) + "\\CMP\\"
            fig, ax = plt.subplots(2,4, figsize=(16,8), dpi=300)
            #fig, ax = plt.subplots(1,1, figsize=(16, 8), dpi=300)
            row_counter    = 0
            column_counter = 0
            counter = 0
            x_tick_locs  = np.linspace(0, S, 6)
            figure_tag   = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
            for sim_id in target_experiments:
                    CMP_nodes_per_step = np.load(CMP_folder + 'experiment_' + str(sim_id) + '\\active_nodes_CMP_' + str(sim_id) + '.npy')
                    CMP_nodes_per_step_ma = pd.rolling_mean(CMP_nodes_per_step, moving_average_period)
                    timeline_CMP = np.arange(0, len(CMP_nodes_per_step), 1)
                    #column_counter = 1
                    #ax[counter].plot(x1,y1, linewidth = 0.05, color = 'k',label = r'$a(t)$')
                    #ax[counter].plot(x1,y2,linewidth = 1.0, color = 'red',label = r'$\langle a(t) \rangle$')
                    #ax[counter].set_xlabel(r'$t \; [ \times 10^{6}]$', fontsize = xaxisfontsize)
                    #ax[counter].set_ylabel(r'$a(t)$', fontsize = yaxisfontsize)
                    #ax[counter].set_xlim([0, S])
                    #ax[counter].set_ylim([0, 5 * L + 1])
                    #ax[counter].set_yticks(range(0, 5 * L + 1, int(1.25 * L)))
                    #ax[counter].set_yticklabels(range(0, 5 * L + 1, int(1.25 * L)))
                    #ax[counter].yaxis.set_label_coords(-0.24, 0.5)
                    #ax[counter].set_xticks(x_tick_locs)
                    #ax[counter].set_xticklabels(x_tick_locs)
                    #ax[counter].tick_params(which='both', width=1.5, labelsize=ticklabelsize)
                    #ax[counter].tick_params(which='major', length=6)
                    #ax[counter].tick_params(which='minor', length=3)
                    #ax[counter].axhline(T,linestyle = 'dashed', color='blue',linewidth = 2.0)

                    ax[row_counter][column_counter].plot(timeline_CMP, CMP_nodes_per_step, linewidth=0.05, color='k', label=r'$a(t)$')
                    ax[row_counter][column_counter].plot(timeline_CMP, CMP_nodes_per_step_ma, linewidth=1.0, color='red', label=r'$\langle a(t) \rangle$')
                    ax[row_counter][column_counter].set_xlabel(r'$t \; [ \times 10^{6}]$', fontsize=xaxisfontsize)
                    ax[row_counter][column_counter].set_ylabel(r'$a(t)$', fontsize=yaxisfontsize)
                    ax[row_counter][column_counter].set_xlim([0, S])
                    ax[row_counter][column_counter].set_ylim([0, 5 * L + 1])
                    ax[row_counter][column_counter].set_yticks(range(0, 5 * L + 1, int(1.25 * L)))
                    ax[row_counter][column_counter].set_yticklabels(range(0, 5 * L + 1, int(1.25 * L)))
                    ax[row_counter][column_counter].yaxis.set_label_coords(-0.24, 0.5)
                    ax[row_counter][column_counter].set_xticks(x_tick_locs)
                    ax[row_counter][column_counter].set_xticklabels(x_tick_locs)
                    ax[row_counter][column_counter].tick_params(which='both', width=1.5, labelsize=ticklabelsize)
                    ax[row_counter][column_counter].tick_params(which='major', length=6)
                    ax[row_counter][column_counter].tick_params(which='minor', length=3)
                    ax[row_counter][column_counter].axhline(T, linestyle='dashed', color='blue', linewidth=2.0)
                    formatter = ScalarFormatter()
                    formatter.set_powerlimits((0,0))
                    
                    #ax[counter].xaxis.set_major_formatter(formatter)
                    #ax[counter].xaxis.get_offset_text().set_fontsize(ticklabelsize)
                    #ax[counter].xaxis.get_offset_text().set_visible(False)
                    #ax[counter].text(-3e5,1.2e3, figure_tag[counter], horizontalalignment='center',verticalalignment = 'center',fontsize=figuretagsize)
                    
                    ax[row_counter][column_counter].xaxis.set_major_formatter(formatter)
                    ax[row_counter][column_counter].xaxis.get_offset_text().set_fontsize(ticklabelsize)
                    ax[row_counter][column_counter].xaxis.get_offset_text().set_visible(False)
                    ax[row_counter][column_counter].text(-3e5, 1.2e3, figure_tag[counter], horizontalalignment='center',verticalalignment='center', fontsize=figuretagsize)
                    counter += 1
                    column_counter = counter%4
                    row_counter    = int(np.floor(counter/4.0))
                    print(counter)
                    '''
                    ax[counter].plot(timeline_CMP, CMP_nodes_per_step, linewidth=0.05, color='k',
                                                         label=r'$a(t)$')
                    ax[counter].plot(timeline_CMP, CMP_nodes_per_step_ma, linewidth=1.0,
                                                         color='red', label=r'$\langle a(t) \rangle$')
                    ax[counter].set_xlabel(r'$t \; [ \times 10^{6}]$', fontsize=xaxisfontsize)
                    ax[counter].set_ylabel(r'$a(t)$', fontsize=yaxisfontsize)
                    ax[counter].set_xlim([0, S])
                    ax[counter].set_ylim([0, 5 * L + 1])
                    ax[counter].set_yticks(range(0, 5 * L + 1, int(1.25 * L)))
                    ax[counter].set_yticklabels(range(0, 5 * L + 1, int(1.25 * L)))
                    ax[counter].yaxis.set_label_coords(-0.24, 0.5)
                    ax[counter].set_xticks(x_tick_locs)
                    ax[counter].set_xticklabels(x_tick_locs)
                    ax[counter].tick_params(which='both', width=1.5, labelsize=ticklabelsize)
                    ax[counter].tick_params(which='major', length=6)
                    ax[counter].tick_params(which='minor', length=3)
                    ax[counter].axhline(T, linestyle='dashed', color='blue', linewidth=2.0)
                    formatter = ScalarFormatter()
                    formatter.set_powerlimits((0, 0))

                    # ax[counter].xaxis.set_major_formatter(formatter)
                    # ax[counter].xaxis.get_offset_text().set_fontsize(ticklabelsize)
                    # ax[counter].xaxis.get_offset_text().set_visible(False)
                    # ax[counter].text(-3e5,1.2e3, figure_tag[counter], horizontalalignment='center',verticalalignment = 'center',fontsize=figuretagsize)

                    ax[counter].xaxis.set_major_formatter(formatter)
                    ax[counter].xaxis.get_offset_text().set_fontsize(ticklabelsize)
                    ax[counter].xaxis.get_offset_text().set_visible(False)
                    ax[counter].text(-3e5, 1.2e3, figure_tag[counter], horizontalalignment='center',
                                                         verticalalignment='center', fontsize=figuretagsize)
                    '''
                    #counter += 1



            #plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
            #plt.tick_params(which='major', length=6)
            #plt.tick_params(which='minor', length=3)
            #plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.rc('font', **{'size': str(ticklabelsize)})
            #plt.subplots_adjust(left=0.25,hspace = 0.5, top = 0.94 - 0.03, bottom = .06 + 0.02)
            plt.subplots_adjust(left=0.06, right = 0.95, wspace = 0.55, hspace=0.5, top=0.94 - 0.03, bottom=.06 + 0.04)
            #plt.tight_layout()
            plt.savefig(figure_path + 'specific_experiments_grid.png')
            plt.close()
