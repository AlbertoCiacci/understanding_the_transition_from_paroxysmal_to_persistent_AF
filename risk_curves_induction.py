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
nu                    = 0.11
min_nu                = 0.05
max_nu                = 0.20
step_nu               = 0.01
tau                   = 50
delta                 = 0.05
delta_array           = [0.05, 0.01, 0.005, 0.0005]
S                     = 1e5
K                     = 200
L                     = 200
T                     = 220
moving_average_period = 220
figure_tag            = 0
xaxisfontsize         = 18
yaxisfontsize         = 16
ticklabelsize         = 16
legendfontsize        = 20
figuretagsize         = 20
confidence_level      = 1.96
########################
#   Functional Filter  #
########################
ma_threshold          = 75
segment_size          = int(1e5)
repeats               = 1
########################
#      Folder          #
########################
#common_path  = "D:\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\mean field model of AF\\Latest version induction\\" + str(delta) + "\\"
#figure_path  = common_path + "figures\\risk curves\\"
#os.makedirs(figure_path, exist_ok=True)
########################
#     Produce Figures  #
########################
'''
stats_cCMP = []
stats_CMP  = []
stats_MF   = []
stats_eMF  = []
stats_cMF  = []
stats_sMF  = []
for i in np.arange(min_nu,max_nu + 0.5*step_nu, step_nu):
           cCMP_enter_AF        = [np.load(common_path + "{:.2f}".format(i) +  '\\cCMP\\experiment_' + str(k) + '\\AF_flag_cCMP_' + str(k) + '.npy').item() for k in range(0, K, 1)]
           CMP_enter_AF         = [np.load(common_path + "{:.2f}".format(i) +  '\\CMP\\experiment_' + str(k) + '\\AF_flag_CMP_' + str(k) + '.npy').item() for k in range(0, K, 1)]
           MF_enter_AF          = [np.load(common_path + "{:.2f}".format(i) + '\\MF\\experiment_' + str(k) + '\\AF_flag_MF_' + str(k) + '.npy').item() for k in range(0, K, 1)]


           cCMP_avg_enter_AF    = np.average(cCMP_enter_AF)
           cCMP_K_tilde         = K + confidence_level ** 2
           cCMP_p_tilde         = (1.0 / float(cCMP_K_tilde)) * (np.sum(cCMP_enter_AF) + 0.5 * (confidence_level ** 2))
           cCMP_ub_enter_AF     = cCMP_p_tilde + confidence_level * np.sqrt((cCMP_p_tilde * (1 - cCMP_p_tilde)) / float(cCMP_K_tilde))
           cCMP_lb_enter_AF     = cCMP_p_tilde - confidence_level * np.sqrt((cCMP_p_tilde * (1 - cCMP_p_tilde)) / float(cCMP_K_tilde))


           CMP_avg_enter_AF     = np.average(CMP_enter_AF)
           CMP_K_tilde          = K + confidence_level ** 2
           CMP_p_tilde          = (1.0 / float(CMP_K_tilde)) * (np.sum(CMP_enter_AF) + 0.5 * (confidence_level ** 2))
           CMP_ub_enter_AF      = CMP_p_tilde + confidence_level * np.sqrt((CMP_p_tilde * (1 - CMP_p_tilde)) / float(CMP_K_tilde))
           CMP_lb_enter_AF      = CMP_p_tilde - confidence_level * np.sqrt((CMP_p_tilde * (1 - CMP_p_tilde)) / float(CMP_K_tilde))


           MF_avg_enter_AF      = np.average(MF_enter_AF)
           MF_K_tilde           = K + confidence_level ** 2
           MF_p_tilde           = (1.0 / float(MF_K_tilde)) * (np.sum(MF_enter_AF) + 0.5 * (confidence_level ** 2))
           MF_ub_enter_AF       = MF_p_tilde + confidence_level * np.sqrt((MF_p_tilde * (1 - MF_p_tilde)) / float(MF_K_tilde))
           MF_lb_enter_AF       = MF_p_tilde - confidence_level * np.sqrt((MF_p_tilde * (1 - MF_p_tilde)) / float(MF_K_tilde))




           stats_cCMP.append([cCMP_avg_enter_AF, cCMP_ub_enter_AF, cCMP_lb_enter_AF])
           stats_CMP.append([CMP_avg_enter_AF, CMP_ub_enter_AF, CMP_lb_enter_AF])

           stats_MF.append([MF_avg_enter_AF, MF_ub_enter_AF, MF_lb_enter_AF])



           print('Data extraction and processing completed. nu: ' + str(i))

x_array                   = np.arange(min_nu, max_nu + 0.5 * step_nu, step_nu)
theory_color              = 'violet'
cCMP_color                = 'red'
CMP_color                 = 'black'
MF_color                  = 'blue'
eMF_color                 = 'orange'
cMF_color                 = 'green'
sMF_color                 = 'brown'
theoretical_curve         = 1.0 - (1 - (1 - x_array) ** tau) ** (delta * L * L)
cCMP_avg_enter_AF_curve   = np.array([stats_cCMP[i][0] for i in range(0, len(x_array),1)])
cCMP_ub_enter_AF_curve    = np.array([stats_cCMP[i][1] for i in range(0, len(x_array),1)])
cCMP_lb_enter_AF_curve    = np.array([stats_cCMP[i][2] for i in range(0, len(x_array),1)])
CMP_avg_enter_AF_curve    = np.array([stats_CMP[i][0] for i in range(0, len(x_array),1)])
CMP_ub_enter_AF_curve     = np.array([stats_CMP[i][1] for i in range(0, len(x_array),1)])
CMP_lb_enter_AF_curve     = np.array([stats_CMP[i][2] for i in range(0, len(x_array),1)])
MF_avg_enter_AF_curve     = np.array([stats_MF[i][0] for i in range(0, len(x_array),1)])
MF_ub_enter_AF_curve      = np.array([stats_MF[i][1] for i in range(0, len(x_array),1)])
MF_lb_enter_AF_curve      = np.array([stats_MF[i][2] for i in range(0, len(x_array),1)])


##########################################################################################
#                         Probability of inducing AF: Theory vs CMP                      #
##########################################################################################
fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)
ax.plot(x_array, theoretical_curve, linewidth=1.5, marker='s', color=theory_color, markersize=4, label= 'Theory')

ax.plot(x_array, CMP_avg_enter_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
ax.plot(x_array, CMP_ub_enter_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
ax.plot(x_array, CMP_lb_enter_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
segments = [[(x_array[ij], CMP_ub_enter_AF_curve[ij]), (x_array[ij], CMP_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
ax.add_collection(lc)


plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
plt.ylabel('Probability of inducing AF', fontsize=yaxisfontsize)
plt.xlim([min_nu, max_nu])
plt.ylim([0, 1.02])
plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu), np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_label_coords(-0.13, 0.5)
plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)
plt.legend(fontsize=legendfontsize)
plt.subplots_adjust(left=0.15)
plt.savefig(figure_path + 'inducing_AF_CMP_vs_Theory.pdf', dpi=300)
plt.close()


##########################################################################################
#                       Probability of inducing AF: cCMP vs CMP vs MF                    #
##########################################################################################
fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)

ax.plot(x_array, cCMP_avg_enter_AF_curve, linewidth=1.5, marker='s', color=cCMP_color, markersize=4, label= 'cCMP')
ax.plot(x_array, cCMP_ub_enter_AF_curve, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
ax.plot(x_array, cCMP_lb_enter_AF_curve, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
segments = [[(x_array[ij], cCMP_ub_enter_AF_curve[ij]), (x_array[ij], cCMP_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=cCMP_color, linewidths=1.5)
ax.add_collection(lc)


ax.plot(x_array, CMP_avg_enter_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
ax.plot(x_array, CMP_ub_enter_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
ax.plot(x_array, CMP_lb_enter_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
segments = [[(x_array[ij], CMP_ub_enter_AF_curve[ij]), (x_array[ij], CMP_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
ax.add_collection(lc)


ax.plot(x_array, MF_avg_enter_AF_curve, linestyle = 'dotted', linewidth=1.5, marker='s', color=MF_color, markersize=4, label= 'MF')
ax.plot(x_array, MF_ub_enter_AF_curve, linewidth=0, marker='_', color=MF_color, markersize=4,label='')
ax.plot(x_array, MF_lb_enter_AF_curve, linewidth=0, marker='_', color=MF_color, markersize=4,label='')
segments = [[(x_array[ij], MF_ub_enter_AF_curve[ij]), (x_array[ij], MF_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=MF_color, linewidths=1.5)
ax.add_collection(lc)


plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
plt.ylabel('Probability of inducing AF', fontsize=yaxisfontsize)
plt.xlim([min_nu, max_nu])
plt.ylim([0, 1.02])
plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu), np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_label_coords(-0.13, 0.5)
ax.text(0.0275, 1.125, '(a)', horizontalalignment='center', verticalalignment='center', fontsize=figuretagsize)
plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)
plt.legend(fontsize=legendfontsize)
plt.subplots_adjust(left=0.15)
plt.savefig(figure_path + 'inducing_AF_cCMP_vs_CMP_vs_MF.pdf', dpi=300)
plt.close()
'''
fig, ax        = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
row_counter    = 0
column_counter = 0
counter        = 0
figure_tag     = ['(a)','(b)','(c)','(d)']
for delta in delta_array:
    common_path = "C:\\Users\\Alberto Ciacci\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\induction analysis short\\" + str(delta) + "\\"
    ########################
    #     Produce Figures  #
    ########################
    stats_cCMP = []
    stats_CMP  = []
    stats_MF   = []
    stats_eMF  = []
    stats_cMF  = []
    stats_sMF  = []
    for i in np.arange(min_nu, max_nu + 0.5 * step_nu, step_nu):
                cCMP_enter_AF = [np.load(common_path + "{:.2f}".format(i) + '\\experiment_' + str(k) + '\\AF_flag_CCMP_' + str(k) + '.npy').item() for k in range(0, K, 1)]
                CMP_enter_AF  = [np.load(common_path + "{:.2f}".format(i) + '\\experiment_' + str(k) + '\\AF_flag_CMP_' + str(k) + '.npy').item() for k in range(0, K, 1)]
                MF_enter_AF  = [np.load(common_path + "{:.2f}".format(i) + '\\experiment_' + str(k) + '\\AF_flag_MF_' + str(k) + '.npy').item()for k in range(0, K, 1)]

                cCMP_avg_enter_AF = np.average(cCMP_enter_AF)
                cCMP_K_tilde = K + confidence_level ** 2
                cCMP_p_tilde = (1.0 / float(cCMP_K_tilde)) * (np.sum(cCMP_enter_AF) + 0.5 * (confidence_level ** 2))
                cCMP_ub_enter_AF = cCMP_p_tilde + confidence_level * np.sqrt((cCMP_p_tilde * (1 - cCMP_p_tilde)) / float(cCMP_K_tilde))
                cCMP_lb_enter_AF = cCMP_p_tilde - confidence_level * np.sqrt((cCMP_p_tilde * (1 - cCMP_p_tilde)) / float(cCMP_K_tilde))

                CMP_avg_enter_AF = np.average(CMP_enter_AF)
                CMP_K_tilde = K + confidence_level ** 2
                CMP_p_tilde = (1.0 / float(CMP_K_tilde)) * (np.sum(CMP_enter_AF) + 0.5 * (confidence_level ** 2))
                CMP_ub_enter_AF = CMP_p_tilde + confidence_level * np.sqrt((CMP_p_tilde * (1 - CMP_p_tilde)) / float(CMP_K_tilde))
                CMP_lb_enter_AF = CMP_p_tilde - confidence_level * np.sqrt((CMP_p_tilde * (1 - CMP_p_tilde)) / float(CMP_K_tilde))

                MF_avg_enter_AF = np.average(MF_enter_AF)
                MF_K_tilde = K + confidence_level ** 2
                MF_p_tilde = (1.0 / float(MF_K_tilde)) * (np.sum(MF_enter_AF) + 0.5 * (confidence_level ** 2))
                MF_ub_enter_AF = MF_p_tilde + confidence_level * np.sqrt((MF_p_tilde * (1 - MF_p_tilde)) / float(MF_K_tilde))
                MF_lb_enter_AF = MF_p_tilde - confidence_level * np.sqrt((MF_p_tilde * (1 - MF_p_tilde)) / float(MF_K_tilde))

                stats_cCMP.append([cCMP_avg_enter_AF, cCMP_ub_enter_AF, cCMP_lb_enter_AF])
                stats_CMP.append([CMP_avg_enter_AF, CMP_ub_enter_AF, CMP_lb_enter_AF])

                stats_MF.append([MF_avg_enter_AF, MF_ub_enter_AF, MF_lb_enter_AF])

    x_array                 = np.arange(min_nu, max_nu + 0.5 * step_nu, step_nu)
    theory_color            = 'violet'
    cCMP_color              = 'red'
    CMP_color               = 'black'
    MF_color                = 'blue'
    eMF_color               = 'orange'
    cMF_color               = 'green'
    sMF_color               = 'brown'
    theoretical_curve       = 1.0 - (1 - (1 - x_array) ** tau) ** (delta * L * L)
    cCMP_avg_enter_AF_curve = np.array([stats_cCMP[i][0] for i in range(0, len(x_array), 1)])
    cCMP_ub_enter_AF_curve  = np.array([stats_cCMP[i][1] for i in range(0, len(x_array), 1)])
    cCMP_lb_enter_AF_curve  = np.array([stats_cCMP[i][2] for i in range(0, len(x_array), 1)])
    CMP_avg_enter_AF_curve  = np.array([stats_CMP[i][0] for i in range(0, len(x_array), 1)])
    CMP_ub_enter_AF_curve   = np.array([stats_CMP[i][1] for i in range(0, len(x_array), 1)])
    CMP_lb_enter_AF_curve   = np.array([stats_CMP[i][2] for i in range(0, len(x_array), 1)])
    MF_avg_enter_AF_curve   = np.array([stats_MF[i][0] for i in range(0, len(x_array), 1)])
    MF_ub_enter_AF_curve    = np.array([stats_MF[i][1] for i in range(0, len(x_array), 1)])
    MF_lb_enter_AF_curve    = np.array([stats_MF[i][2] for i in range(0, len(x_array), 1)])

    ax[row_counter][column_counter].plot(x_array, cCMP_avg_enter_AF_curve, linewidth=1.5, marker='s', color=cCMP_color, markersize=4, label='cCMP')
    ax[row_counter][column_counter].plot(x_array, cCMP_ub_enter_AF_curve, linewidth=0, marker='_', color=cCMP_color, markersize=4, label='')
    ax[row_counter][column_counter].plot(x_array, cCMP_lb_enter_AF_curve, linewidth=0, marker='_', color=cCMP_color, markersize=4, label='')
    segments = [[(x_array[ij], cCMP_ub_enter_AF_curve[ij]), (x_array[ij], cCMP_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
    lc = mc.LineCollection(segments, colors=cCMP_color, linewidths=1.5)
    ax[row_counter][column_counter].add_collection(lc)

    ax[row_counter][column_counter].plot(x_array, CMP_avg_enter_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label='CMP')
    ax[row_counter][column_counter].plot(x_array, CMP_ub_enter_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4, label='')
    ax[row_counter][column_counter].plot(x_array, CMP_lb_enter_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4, label='')
    segments = [[(x_array[ij], CMP_ub_enter_AF_curve[ij]), (x_array[ij], CMP_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
    lc = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
    ax[row_counter][column_counter].add_collection(lc)

    '''
    ax[row_counter][column_counter].plot(x_array, MF_avg_enter_AF_curve, linestyle='dotted', linewidth=1.5, marker='s', color=MF_color, markersize=4, label='MF')
    ax[row_counter][column_counter].plot(x_array, MF_ub_enter_AF_curve, linewidth=0, marker='_', color=MF_color, markersize=4, label='')
    ax[row_counter][column_counter].plot(x_array, MF_lb_enter_AF_curve, linewidth=0, marker='_', color=MF_color, markersize=4, label='')
    segments = [[(x_array[ij], MF_ub_enter_AF_curve[ij]), (x_array[ij], MF_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
    lc = mc.LineCollection(segments, colors=MF_color, linewidths=1.5)
    ax[row_counter][column_counter].add_collection(lc)
    '''

    if row_counter == 1:
        ax[row_counter][column_counter].set_xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
        ax[row_counter][column_counter].tick_params(axis = 'x', which='both', width=1.5, labelsize=ticklabelsize)
    else:
        ax[row_counter][column_counter].tick_params(axis='x', which='both', width=1.5, labelsize=0)
    if column_counter == 0:
        ax[row_counter][column_counter].set_ylabel('Probability of inducing AF', fontsize=yaxisfontsize)
        ax[row_counter][column_counter].tick_params(axis='y', which='both', width=1.5, labelsize=ticklabelsize)
    else:
        ax[row_counter][column_counter].tick_params(axis='y', which='both', width=1.5, labelsize=0)
    ax[row_counter][column_counter].set_xlim([min_nu, max_nu])
    ax[row_counter][column_counter].set_ylim([0, 1.02])
    ax[row_counter][column_counter].set_xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
    ax[row_counter][column_counter].set_xticklabels(np.round(np.arange(min_nu, max_nu + step_nu, 5 * step_nu),2))
    ax[row_counter][column_counter].xaxis.set_minor_locator(AutoMinorLocator(5))
    ax[row_counter][column_counter].yaxis.set_label_coords(-0.30, 0.5)
    ax[row_counter][column_counter].text(0.025, 1.125, figure_tag[counter], horizontalalignment='center', verticalalignment='center', fontsize=figuretagsize)
    ax[row_counter][column_counter].text(0.17, 0.95, r'$\delta = $' + str(delta), horizontalalignment='center', verticalalignment='center', fontsize=13)
    #ax[row_counter][column_counter].tick_params(which='both', width=1.5, labelsize=ticklabelsize)
    ax[row_counter][column_counter].tick_params(which='major', length=6)
    ax[row_counter][column_counter].tick_params(which='minor', length=3)
    #plt.legend(fontsize=legendfontsize)
    fig.subplots_adjust(left=0.15)

    counter += 1
    column_counter = counter % 2
    row_counter = int(np.floor(counter / 2.0))

plt.subplots_adjust(left=0.15, right = 0.95, wspace = 0.25, hspace=0.25, top=0.94 - 0.03, bottom=.06 + 0.04)
figure_path = "C:\\Users\\Alberto Ciacci\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\induction analysis short\\"
plt.savefig(figure_path + 'inducing_AF_grid.pdf')
plt.close()


