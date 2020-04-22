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
min_nu                = 0.05
max_nu                = 0.20
step_nu               = 0.01
tau                   = 50
delta                 = 0.01
S                     = 1e6
K                     = 200
L                     = 200
T                     = 220
moving_average_period = 220
figure_tag            = 0
xaxisfontsize         = 20
yaxisfontsize         = 18
ticklabelsize         = 16
legendfontsize        = 20
figuretagsize         = 20
confidence_level      = 1.96
add_MF                = True
########################
#   Functional Filter  #
########################
ma_threshold          = 150
segment_size          = int(1e5)
repeats               = 50
########################
#      Folder          #
########################
common_path  = "D:\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\risk analysis\\"
figure_path  = common_path + "figures\\risk curves test III\\"
os.makedirs(figure_path, exist_ok=True)
########################
#     Produce Figures  #
########################
stats_cCMP = []
stats_CMP  = []
stats_MF   = []
stats_eMF  = []
stats_cMF  = []
stats_sMF  = []
N_vs_risk_cCMP     = [[] for j in range(0,70,1)]
N_vs_risk_MF       = [[] for j in range(0,70,1)]
nu_vs_cs           = []
N_vs_duration_cCMP = [[] for j in range(0,70,1)]
N_vs_duration_MF   = [[] for j in range(0,70,1)]
for i in np.arange(min_nu,max_nu + 0.5*step_nu, step_nu):
           cCMP_time_in_AF      = [np.load(common_path + "{:.2f}".format(i) +  '\\cCMP\\experiment_' + str(k) + '\\af_risk_cCMP_' + str(k) + '.npy').item() for k in range(0, K, 1)]
           cCMP_enter_AF        = [np.load(common_path + "{:.2f}".format(i) +  '\\cCMP\\experiment_' + str(k) + '\\af_flag_cCMP_' + str(k) + '.npy').item() for k in range(0, K, 1)]
           cCMP_critical_struc  = [len(np.load(common_path + "{:.2f}".format(i) +  '\\cCMP\\experiment_' + str(k) + '\\cs_length_cCMP_' + str(k) + '.npy')) for k in range(0, K, 1)]
           cCMP_durations       = [mtd.cmp_durations(np.load(common_path + "{:.2f}".format(i) + '\\cCMP\\experiment_' + str(k) + '\\active_nodes_cCMP_' + str(k) + '.npy') >= T) for k in range(0, K, 1)]
           CMP_time_in_AF       = [np.load(common_path + "{:.2f}".format(i) +  '\\CMP\\experiment_' + str(k) + '\\af_risk_CMP_' + str(k) + '.npy').item() for k in range(0, K, 1)]
           CMP_enter_AF         = [np.load(common_path + "{:.2f}".format(i) +  '\\CMP\\experiment_' + str(k) + '\\af_flag_CMP_' + str(k) + '.npy').item() for k in range(0, K, 1)]
           if add_MF:
                   MF_time_in_AF        = [np.load(common_path + "{:.2f}".format(i) + '\\MF\\experiment_' + str(k) + '\\af_risk_MF_' + str(k) + '.npy').item() for k in range(0, K, 1)]
                   MF_enter_AF          = [np.load(common_path + "{:.2f}".format(i) + '\\MF\\experiment_' + str(k) + '\\af_flag_MF_' + str(k) + '.npy').item() for k in range(0, K, 1)]
                   MF_durations         = [mtd.cmp_durations(np.load(common_path + "{:.2f}".format(i) + '\\MF\\experiment_' + str(k) + '\\active_particles_MF' + str(k) + '.npy') >= 1) for k in range(0, K, 1)]
                   eMF_time_in_AF       = [np.load(common_path + "{:.2f}".format(i) + '\\eMF\\experiment_' + str(k) + '\\af_risk_eMF_' + str(k) + '.npy').item() for k in range(0, K, 1)]
                   cMF_time_in_AF       = [np.load(common_path + "{:.2f}".format(i) + '\\cMF\\experiment_' + str(k) + '\\af_risk_cMF_' + str(k) + '.npy').item() for k in range(0, K, 1)]
                   sMF_time_in_AF       = [np.load(common_path + "{:.2f}".format(i) + '\\2MF\\experiment_' + str(k) + '\\af_risk_2MF_' + str(k) + '.npy').item() for k in range(0, K, 1)]




           cCMP_avg_time_in_AF  = np.average(cCMP_time_in_AF)
           cCMP_std_time_in_AF  = np.std(cCMP_time_in_AF)
           cCMP_ub_time_in_AF   = cCMP_avg_time_in_AF + confidence_level * (cCMP_std_time_in_AF / float(np.sqrt(K)))
           cCMP_lb_time_in_AF   = cCMP_avg_time_in_AF - confidence_level * (cCMP_std_time_in_AF / float(np.sqrt(K)))
           cCMP_avg_enter_AF    = np.average(cCMP_enter_AF)
           cCMP_K_tilde         = K + confidence_level ** 2
           cCMP_p_tilde         = (1.0 / float(cCMP_K_tilde)) * (np.sum(cCMP_enter_AF) + 0.5 * (confidence_level ** 2))
           cCMP_ub_enter_AF     = cCMP_p_tilde + confidence_level * np.sqrt((cCMP_p_tilde * (1 - cCMP_p_tilde)) / float(cCMP_K_tilde))
           cCMP_lb_enter_AF     = cCMP_p_tilde - confidence_level * np.sqrt((cCMP_p_tilde * (1 - cCMP_p_tilde)) / float(cCMP_K_tilde))

           CMP_avg_time_in_AF   = np.average(CMP_time_in_AF)
           CMP_std_time_in_AF   = np.std(CMP_time_in_AF)
           CMP_ub_time_in_AF    = CMP_avg_time_in_AF + confidence_level * (CMP_std_time_in_AF / float(np.sqrt(K)))
           CMP_lb_time_in_AF    = CMP_avg_time_in_AF - confidence_level * (CMP_std_time_in_AF / float(np.sqrt(K)))
           CMP_avg_enter_AF     = np.average(CMP_enter_AF)
           CMP_K_tilde          = K + confidence_level ** 2
           CMP_p_tilde          = (1.0 / float(CMP_K_tilde)) * (np.sum(CMP_enter_AF) + 0.5 * (confidence_level ** 2))
           CMP_ub_enter_AF      = CMP_p_tilde + confidence_level * np.sqrt((CMP_p_tilde * (1 - CMP_p_tilde)) / float(CMP_K_tilde))
           CMP_lb_enter_AF      = CMP_p_tilde - confidence_level * np.sqrt((CMP_p_tilde * (1 - CMP_p_tilde)) / float(CMP_K_tilde))

           if add_MF:

                   MF_avg_time_in_AF    = np.average(MF_time_in_AF)
                   MF_std_time_in_AF    = np.std(MF_time_in_AF)
                   MF_ub_time_in_AF     = MF_avg_time_in_AF + confidence_level * (MF_std_time_in_AF / float(np.sqrt(K)))
                   MF_lb_time_in_AF     = MF_avg_time_in_AF - confidence_level * (MF_std_time_in_AF / float(np.sqrt(K)))
                   MF_avg_enter_AF      = np.average(MF_enter_AF)
                   MF_K_tilde           = K + confidence_level ** 2
                   MF_p_tilde           = (1.0 / float(MF_K_tilde)) * (np.sum(MF_enter_AF) + 0.5 * (confidence_level ** 2))
                   MF_ub_enter_AF       = MF_p_tilde + confidence_level * np.sqrt((MF_p_tilde * (1 - MF_p_tilde)) / float(MF_K_tilde))
                   MF_lb_enter_AF       = MF_p_tilde - confidence_level * np.sqrt((MF_p_tilde * (1 - MF_p_tilde)) / float(MF_K_tilde))

                   eMF_avg_time_in_AF   = np.average(eMF_time_in_AF)
                   eMF_std_time_in_AF   = np.std(eMF_time_in_AF)
                   eMF_ub_time_in_AF    = eMF_avg_time_in_AF + confidence_level * (eMF_std_time_in_AF / float(np.sqrt(K)))
                   eMF_lb_time_in_AF    = eMF_avg_time_in_AF - confidence_level * (eMF_std_time_in_AF / float(np.sqrt(K)))

                   cMF_avg_time_in_AF   = np.average(cMF_time_in_AF)
                   cMF_std_time_in_AF   = np.std(cMF_time_in_AF)
                   cMF_ub_time_in_AF    = cMF_avg_time_in_AF + confidence_level * (cMF_std_time_in_AF / float(np.sqrt(K)))
                   cMF_lb_time_in_AF    = cMF_avg_time_in_AF - confidence_level * (cMF_std_time_in_AF / float(np.sqrt(K)))

                   sMF_avg_time_in_AF   = np.average(sMF_time_in_AF)
                   sMF_std_time_in_AF   = np.std(sMF_time_in_AF)
                   sMF_ub_time_in_AF    = sMF_avg_time_in_AF + confidence_level * (sMF_std_time_in_AF / float(np.sqrt(K)))
                   sMF_lb_time_in_AF    = sMF_avg_time_in_AF - confidence_level * (sMF_std_time_in_AF / float(np.sqrt(K)))


           stats_cCMP.append([cCMP_avg_time_in_AF, cCMP_ub_time_in_AF, cCMP_lb_time_in_AF, cCMP_avg_enter_AF, cCMP_ub_enter_AF, cCMP_lb_enter_AF])
           stats_CMP.append([CMP_avg_time_in_AF, CMP_ub_time_in_AF, CMP_lb_time_in_AF, CMP_avg_enter_AF, CMP_ub_enter_AF, CMP_lb_enter_AF])

           if add_MF:
                   stats_MF.append([MF_avg_time_in_AF, MF_ub_time_in_AF, MF_lb_time_in_AF, MF_avg_enter_AF, MF_ub_enter_AF, MF_lb_enter_AF])
                   stats_eMF.append([eMF_avg_time_in_AF, eMF_ub_time_in_AF, eMF_lb_time_in_AF])
                   stats_cMF.append([cMF_avg_time_in_AF, cMF_ub_time_in_AF, cMF_lb_time_in_AF])
                   stats_sMF.append([sMF_avg_time_in_AF, sMF_ub_time_in_AF, sMF_lb_time_in_AF])


           for j in range(len(cCMP_critical_struc)):
               N_vs_risk_cCMP[cCMP_critical_struc[j]].append(cCMP_time_in_AF[j])
               N_vs_duration_cCMP[cCMP_critical_struc[j]].extend(cCMP_durations[j])
               if add_MF:
                   N_vs_risk_MF[cCMP_critical_struc[j]].append(MF_time_in_AF[j])
                   N_vs_duration_MF[cCMP_critical_struc[j]].extend(MF_durations[j])

           nu_vs_cs.append(cCMP_critical_struc)


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
cCMP_avg_time_in_AF_curve = np.array([stats_cCMP[i][0] for i in range(0, len(x_array),1)])
cCMP_ub_time_in_AF_curve  = np.array([stats_cCMP[i][1] for i in range(0, len(x_array),1)])
cCMP_lb_time_in_AF_curve  = np.array([stats_cCMP[i][2] for i in range(0, len(x_array),1)])
cCMP_avg_enter_AF_curve   = np.array([stats_cCMP[i][3] for i in range(0, len(x_array),1)])
cCMP_ub_enter_AF_curve    = np.array([stats_cCMP[i][4] for i in range(0, len(x_array),1)])
cCMP_lb_enter_AF_curve    = np.array([stats_cCMP[i][5] for i in range(0, len(x_array),1)])
CMP_avg_time_in_AF_curve  = np.array([stats_CMP[i][0] for i in range(0, len(x_array),1)])
CMP_ub_time_in_AF_curve   = np.array([stats_CMP[i][1] for i in range(0, len(x_array),1)])
CMP_lb_time_in_AF_curve   = np.array([stats_CMP[i][2] for i in range(0, len(x_array),1)])
CMP_avg_enter_AF_curve    = np.array([stats_CMP[i][3] for i in range(0, len(x_array),1)])
CMP_ub_enter_AF_curve     = np.array([stats_CMP[i][4] for i in range(0, len(x_array),1)])
CMP_lb_enter_AF_curve     = np.array([stats_CMP[i][5] for i in range(0, len(x_array),1)])

if add_MF:
        MF_avg_time_in_AF_curve   = np.array([stats_MF[i][0] for i in range(0, len(x_array),1)])
        MF_ub_time_in_AF_curve    = np.array([stats_MF[i][1] for i in range(0, len(x_array),1)])
        MF_lb_time_in_AF_curve    = np.array([stats_MF[i][2] for i in range(0, len(x_array),1)])
        MF_avg_enter_AF_curve     = np.array([stats_MF[i][3] for i in range(0, len(x_array),1)])
        MF_ub_enter_AF_curve      = np.array([stats_MF[i][4] for i in range(0, len(x_array),1)])
        MF_lb_enter_AF_curve      = np.array([stats_MF[i][5] for i in range(0, len(x_array),1)])
        eMF_avg_time_in_AF_curve  = np.array([stats_eMF[i][0] for i in range(0, len(x_array),1)])
        eMF_ub_time_in_AF_curve   = np.array([stats_eMF[i][1] for i in range(0, len(x_array),1)])
        eMF_lb_time_in_AF_curve   = np.array([stats_eMF[i][2] for i in range(0, len(x_array),1)])
        cMF_avg_time_in_AF_curve  = np.array([stats_cMF[i][0] for i in range(0, len(x_array),1)])
        cMF_ub_time_in_AF_curve   = np.array([stats_cMF[i][1] for i in range(0, len(x_array),1)])
        cMF_lb_time_in_AF_curve   = np.array([stats_cMF[i][2] for i in range(0, len(x_array),1)])
        sMF_avg_time_in_AF_curve  = np.array([stats_sMF[i][0] for i in range(0, len(x_array),1)])
        sMF_ub_time_in_AF_curve   = np.array([stats_sMF[i][1] for i in range(0, len(x_array),1)])
        sMF_lb_time_in_AF_curve   = np.array([stats_sMF[i][2] for i in range(0, len(x_array),1)])

##########################################################################################
#                         N vs risk                                                      #
##########################################################################################
fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)
x_N                       = np.arange(0,70,1)
y_cCMP_avg                = [np.nanmean(r) for r in N_vs_risk_cCMP]
y_cCMP_ub                 = [np.nanmean(r) + confidence_level*(np.std(r)/float(np.sqrt(len(r)))) for r in N_vs_risk_cCMP]
y_cCMP_lb                 = [np.nanmean(r) - confidence_level*(np.std(r)/float(np.sqrt(len(r)))) for r in N_vs_risk_cCMP]
if add_MF:
            y_MF_avg   = [np.nanmean(r) for r in N_vs_risk_MF]
            y_MF_ub    = [np.nanmean(r) + confidence_level * (np.std(r) / float(np.sqrt(len(r)))) for r in N_vs_risk_MF]
            y_MF_lb    = [np.nanmean(r) - confidence_level * (np.std(r) / float(np.sqrt(len(r)))) for r in N_vs_risk_MF]
            ax.plot(x_N, y_MF_avg, linewidth=1.5, marker='s', color=MF_color, markersize=4, label='MF')
            ax.plot(x_N, y_MF_ub, linewidth=0, marker='_', color=MF_color, markersize=4, label='')
            ax.plot(x_N, y_MF_lb, linewidth=0, marker='_', color=MF_color, markersize=4, label='')
ax.plot(x_N, y_cCMP_avg, linewidth=1.5, marker='s', color=cCMP_color, markersize=4, label= 'cCMP')
ax.plot(x_N, y_cCMP_ub, linewidth=0, marker='_', color=cCMP_color, markersize=4, label='')
ax.plot(x_N, y_cCMP_lb, linewidth=0, marker='_', color=cCMP_color, markersize=4, label='')
'''
segments = [[(x_array[ij], CMP_ub_enter_AF_curve[ij]), (x_array[ij], CMP_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
ax.add_collection(lc)


            idx_1       = np.where(np.array(CMP_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(CMP_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = CMP_avg_time_in_AF_curve[idx_1]
            y_2         = CMP_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)
'''
ax.hlines(y = 0.5, xmin = 0, xmax = 2,linewidth=1.25, linestyle=(0, (5, 5)))
ax.vlines(x = 2, ymin = 0, ymax = 0.5,linewidth=1.25, linestyle=(0, (5, 5)))
plt.xlabel(r'$N$', fontsize=xaxisfontsize)
plt.ylabel('Time in AF', fontsize=yaxisfontsize)
plt.ylim([0, 1.02])
plt.xlim([0, 10])
plt.xticks(np.arange(0,10,2), np.arange(0,10,2))
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_label_coords(-0.13, 0.5)
plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)
plt.legend(fontsize=legendfontsize)
plt.subplots_adjust(left=0.15)
plt.savefig(figure_path + 'N_vs_risk_cCMP_vs_MF.pdf', dpi=300)
plt.close()

##########################################################################################
#                         nu vs N                                                        #
##########################################################################################
fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)
y_cCMP_avg                = np.array([np.nanmean(r) for r in nu_vs_cs])
#y_cCMP_ub                 = [np.nanmean(r) + confidence_level*(np.std(r)/float(np.sqrt(len(r)))) for r in nu_vs_cs]
#y_cCMP_lb                 = [np.nanmean(r) - confidence_level*(np.std(r)/float(np.sqrt(len(r)))) for r in nu_vs_cs]
y_cCMP_ub                 = [np.nanmean(r) + confidence_level*(np.std(r)) for r in nu_vs_cs]
y_cCMP_lb                 = [np.nanmean(r) - confidence_level*(np.std(r)) for r in nu_vs_cs]
ax.plot(x_array, y_cCMP_avg, linewidth=1.5, marker='s', color=CMP_color, markersize=4)
ax.plot(x_array, y_cCMP_ub, linewidth=0, marker='_', color=CMP_color, markersize=4, label='')
ax.plot(x_array, y_cCMP_lb, linewidth=0, marker='_', color=CMP_color, markersize=4, label='')
segments = [[(x_array[ij], y_cCMP_ub [ij]), (x_array[ij], y_cCMP_lb[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors='k', linewidths=1.0)
ax.add_collection(lc)
ax.hlines(y = 2, xmin = min_nu, xmax = 0.1,linewidth=1.25, linestyle=(0, (5, 5)))
ax.vlines(x = 0.1, ymin = 0, ymax = 2,linewidth=1.25, linestyle=(0, (5, 5)))
'''
ax.plot(x_array, CMP_avg_enter_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
ax.plot(x_array, CMP_ub_enter_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
ax.plot(x_array, CMP_lb_enter_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
segments = [[(x_array[ij], CMP_ub_enter_AF_curve[ij]), (x_array[ij], CMP_lb_enter_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
ax.add_collection(lc)
'''

plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
plt.ylabel(r'$N$', fontsize=yaxisfontsize)
plt.xlim([min_nu - 0.0025, max_nu])
plt.ylim([0, 37])
plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu), np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_label_coords(-0.13, 0.5)
plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)
plt.legend(fontsize=legendfontsize)
plt.subplots_adjust(left=0.15)
plt.savefig(figure_path + 'nu_vs_N.pdf', dpi=300)
plt.close()

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


if add_MF:
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


if add_MF:
            ##########################################################################################
            #                             Time in AF: CMP vs MF                                      #
            ##########################################################################################
            fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)

            ax.plot(x_array, CMP_avg_time_in_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
            ax.plot(x_array, CMP_ub_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
            ax.plot(x_array, CMP_lb_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
            segments = [[(x_array[ij], CMP_ub_time_in_AF_curve[ij]), (x_array[ij], CMP_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
            ax.add_collection(lc)

            ax.plot(x_array, MF_avg_time_in_AF_curve, linewidth=1.5, marker='s', color= MF_color, markersize=4, label= 'MF')
            ax.plot(x_array, MF_ub_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
            ax.plot(x_array, MF_lb_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
            segments = [[(x_array[ij], MF_ub_time_in_AF_curve[ij]), (x_array[ij], MF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors= MF_color, linewidths=1.5)
            ax.add_collection(lc)


            idx_1       = np.where(np.array(CMP_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(CMP_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = CMP_avg_time_in_AF_curve[idx_1]
            y_2         = CMP_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            idx_1       = np.where(np.array(MF_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(MF_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = MF_avg_time_in_AF_curve[idx_1]
            y_2         = MF_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=MF_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
            plt.ylabel('Time in AF', fontsize=yaxisfontsize)
            plt.xlim([min_nu, max_nu])
            plt.ylim([0, 1.02])
            plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu),np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_label_coords(-0.13, 0.5)
            plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
            plt.tick_params(which='major', length=6)
            plt.tick_params(which='minor', length=3)
            plt.legend(fontsize=legendfontsize)
            plt.subplots_adjust(left=0.15)
            plt.savefig(figure_path + 'time_in_AF_CMP_vs_MF.pdf', dpi=300)
            plt.close()

##########################################################################################
#                             Time in AF: cCMP vs CMP vs MF                              #
##########################################################################################


fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)
ax.plot(x_array, cCMP_avg_time_in_AF_curve, linewidth=1.5, marker='s', color=cCMP_color, markersize=4, label= 'cCMP')
ax.plot(x_array, cCMP_ub_time_in_AF_curve, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
ax.plot(x_array, cCMP_lb_time_in_AF_curve, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
segments = [[(x_array[ij], cCMP_ub_time_in_AF_curve[ij]), (x_array[ij], cCMP_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=cCMP_color, linewidths=1.5)
ax.add_collection(lc)

ax.plot(x_array, CMP_avg_time_in_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
ax.plot(x_array, CMP_ub_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
ax.plot(x_array, CMP_lb_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
segments = [[(x_array[ij], CMP_ub_time_in_AF_curve[ij]), (x_array[ij], CMP_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
ax.add_collection(lc)

if add_MF:
        ax.plot(x_array, MF_avg_time_in_AF_curve, linewidth=1.5, marker='s', color= MF_color, markersize=4, label= 'MF')
        ax.plot(x_array, MF_ub_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
        ax.plot(x_array, MF_lb_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
        segments = [[(x_array[ij], MF_ub_time_in_AF_curve[ij]), (x_array[ij], MF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
        lc       = mc.LineCollection(segments, colors= MF_color, linewidths=1.5)
        ax.add_collection(lc)

idx_1       = np.where(np.array(cCMP_avg_time_in_AF_curve) >= 0.5)[0][-1]
idx_2       = np.where(np.array(cCMP_avg_time_in_AF_curve) < 0.5)[0][0]
y_1         = cCMP_avg_time_in_AF_curve[idx_1]
y_2         = cCMP_avg_time_in_AF_curve[idx_2]
nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
transitions = [[(nu_star, 0), (nu_star, 0.5)]]
lc          = mc.LineCollection(transitions, colors=cCMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
ax.add_collection(lc)

idx_1       = np.where(np.array(CMP_avg_time_in_AF_curve) >= 0.5)[0][-1]
idx_2       = np.where(np.array(CMP_avg_time_in_AF_curve) < 0.5)[0][0]
y_1         = CMP_avg_time_in_AF_curve[idx_1]
y_2         = CMP_avg_time_in_AF_curve[idx_2]
nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
transitions = [[(nu_star, 0), (nu_star, 0.5)]]
lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
ax.add_collection(lc)

if add_MF:
        idx_1       = np.where(np.array(MF_avg_time_in_AF_curve) >= 0.5)[0][-1]
        idx_2       = np.where(np.array(MF_avg_time_in_AF_curve) < 0.5)[0][0]
        y_1         = MF_avg_time_in_AF_curve[idx_1]
        y_2         = MF_avg_time_in_AF_curve[idx_2]
        nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
        transitions = [[(nu_star, 0), (nu_star, 0.5)]]
        lc          = mc.LineCollection(transitions, colors=MF_color, linewidths=1.25, linestyle=(0, (5, 10)))
        ax.add_collection(lc)

plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
plt.ylabel('Time in AF', fontsize=yaxisfontsize)
plt.xlim([min_nu, max_nu])
plt.ylim([0, 1.02])
plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu),np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_label_coords(-0.13, 0.5)
ax.text(0.0275, 1.125, '(b)', horizontalalignment='center', verticalalignment='center',fontsize=figuretagsize)
plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)
plt.legend(fontsize=legendfontsize)
plt.subplots_adjust(left=0.15)
plt.savefig(figure_path + 'time_in_AF_cCMP_vs_CMP_vs_MF.pdf', dpi=300)
plt.close()

if add_MF:
            ##########################################################################################
            #                             Time in AF: CMP vs MF vs eMF                              #
            ##########################################################################################

            fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)

            ax.plot(x_array, CMP_avg_time_in_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
            ax.plot(x_array, CMP_ub_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
            ax.plot(x_array, CMP_lb_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
            segments = [[(x_array[ij], CMP_ub_time_in_AF_curve[ij]), (x_array[ij], CMP_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
            ax.add_collection(lc)

            ax.plot(x_array, MF_avg_time_in_AF_curve, linewidth=1.5, marker='s', color= MF_color, markersize=4, label= 'MF')
            ax.plot(x_array, MF_ub_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
            ax.plot(x_array, MF_lb_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
            segments = [[(x_array[ij], MF_ub_time_in_AF_curve[ij]), (x_array[ij], MF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors= MF_color, linewidths=1.5)
            ax.add_collection(lc)

            ax.plot(x_array, eMF_avg_time_in_AF_curve, linestyle = 'dotted',linewidth=1.5, marker='s', color= eMF_color, markersize=4, label= 'eMF')
            ax.plot(x_array, eMF_ub_time_in_AF_curve,linewidth=0, marker='_', color= eMF_color, markersize=4,label='')
            ax.plot(x_array, eMF_lb_time_in_AF_curve, linewidth=0, marker='_', color= eMF_color, markersize=4,label='')
            segments = [[(x_array[ij], eMF_ub_time_in_AF_curve[ij]), (x_array[ij], eMF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors= eMF_color, linestyle = 'dotted', linewidths=1.5)
            ax.add_collection(lc)


            idx_1       = np.where(np.array(CMP_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(CMP_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = CMP_avg_time_in_AF_curve[idx_1]
            y_2         = CMP_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            idx_1       = np.where(np.array(MF_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(MF_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = MF_avg_time_in_AF_curve[idx_1]
            y_2         = MF_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=MF_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            idx_1       = np.where(np.array(eMF_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(eMF_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = eMF_avg_time_in_AF_curve[idx_1]
            y_2         = eMF_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=eMF_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
            plt.ylabel('Time in AF', fontsize=yaxisfontsize)
            plt.xlim([min_nu, max_nu])
            plt.ylim([0, 1.02])
            plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu),np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_label_coords(-0.13, 0.5)
            plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
            plt.tick_params(which='major', length=6)
            plt.tick_params(which='minor', length=3)
            plt.legend(fontsize=legendfontsize)
            plt.subplots_adjust(left=0.15)
            plt.savefig(figure_path + 'time_in_AF_CMP_vs_MF_vs_eMF.pdf', dpi=300)
            plt.close()

            ##########################################################################################
            #                             Time in AF: CMP vs MF vs cMF                              #
            ##########################################################################################

            fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)

            ax.plot(x_array, CMP_avg_time_in_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
            ax.plot(x_array, CMP_ub_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
            ax.plot(x_array, CMP_lb_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
            segments = [[(x_array[ij], CMP_ub_time_in_AF_curve[ij]), (x_array[ij], CMP_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
            ax.add_collection(lc)

            ax.plot(x_array, MF_avg_time_in_AF_curve, linewidth=1.5, marker='s', color= MF_color, markersize=4, label= 'MF')
            ax.plot(x_array, MF_ub_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
            ax.plot(x_array, MF_lb_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
            segments = [[(x_array[ij], MF_ub_time_in_AF_curve[ij]), (x_array[ij], MF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors= MF_color, linewidths=1.5)
            ax.add_collection(lc)

            ax.plot(x_array, cMF_avg_time_in_AF_curve, linestyle = 'dotted', linewidth=1.5, marker='s', color= cMF_color, markersize=4, label= 'cMF')
            ax.plot(x_array, cMF_ub_time_in_AF_curve, linewidth=0, marker='_', color= cMF_color, markersize=4,label='')
            ax.plot(x_array, cMF_lb_time_in_AF_curve, linewidth=0, marker='_', color= cMF_color, markersize=4,label='')
            segments = [[(x_array[ij], cMF_ub_time_in_AF_curve[ij]), (x_array[ij], cMF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors= cMF_color, linestyle = 'dotted', linewidths=1.5)
            ax.add_collection(lc)


            idx_1       = np.where(np.array(CMP_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(CMP_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = CMP_avg_time_in_AF_curve[idx_1]
            y_2         = CMP_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            idx_1       = np.where(np.array(MF_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(MF_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = MF_avg_time_in_AF_curve[idx_1]
            y_2         = MF_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=MF_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            idx_1       = np.where(np.array(cMF_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(cMF_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = cMF_avg_time_in_AF_curve[idx_1]
            y_2         = cMF_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=cMF_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
            plt.ylabel('Time in AF', fontsize=yaxisfontsize)
            plt.xlim([min_nu, max_nu])
            plt.ylim([0, 1.02])
            plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu),np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_label_coords(-0.13, 0.5)
            plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
            plt.tick_params(which='major', length=6)
            plt.tick_params(which='minor', length=3)
            plt.legend(fontsize=legendfontsize)
            plt.subplots_adjust(left=0.15)
            plt.savefig(figure_path + 'time_in_AF_CMP_vs_MF_vs_cMF.pdf', dpi=300)
            plt.close()

            ##########################################################################################
            #                             Time in AF: CMP vs MF vs 2MF                              #
            ##########################################################################################

            fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)

            ax.plot(x_array, CMP_avg_time_in_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
            ax.plot(x_array, CMP_ub_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
            ax.plot(x_array, CMP_lb_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
            segments = [[(x_array[ij], CMP_ub_time_in_AF_curve[ij]), (x_array[ij], CMP_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
            ax.add_collection(lc)

            ax.plot(x_array, MF_avg_time_in_AF_curve, linewidth=1.5, marker='s', color= MF_color, markersize=4, label= 'MF')
            ax.plot(x_array, MF_ub_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
            ax.plot(x_array, MF_lb_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
            segments = [[(x_array[ij], MF_ub_time_in_AF_curve[ij]), (x_array[ij], MF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors= MF_color, linewidths=1.5)
            ax.add_collection(lc)

            ax.plot(x_array, sMF_avg_time_in_AF_curve, linewidth=1.5, marker='s', color= sMF_color, markersize=4, label= '2MF')
            ax.plot(x_array, sMF_ub_time_in_AF_curve, linewidth=0, marker='_', color= sMF_color, markersize=4,label='')
            ax.plot(x_array, sMF_lb_time_in_AF_curve, linewidth=0, marker='_', color= sMF_color, markersize=4,label='')
            segments = [[(x_array[ij], sMF_ub_time_in_AF_curve[ij]), (x_array[ij], sMF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
            lc       = mc.LineCollection(segments, colors= sMF_color, linewidths=1.5)
            ax.add_collection(lc)


            idx_1       = np.where(np.array(CMP_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(CMP_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = CMP_avg_time_in_AF_curve[idx_1]
            y_2         = CMP_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            idx_1       = np.where(np.array(MF_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(MF_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = MF_avg_time_in_AF_curve[idx_1]
            y_2         = MF_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=MF_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            idx_1       = np.where(np.array(sMF_avg_time_in_AF_curve) >= 0.5)[0][-1]
            idx_2       = np.where(np.array(sMF_avg_time_in_AF_curve) < 0.5)[0][0]
            y_1         = sMF_avg_time_in_AF_curve[idx_1]
            y_2         = sMF_avg_time_in_AF_curve[idx_2]
            nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
            transitions = [[(nu_star, 0), (nu_star, 0.5)]]
            lc          = mc.LineCollection(transitions, colors=sMF_color, linewidths=1.25, linestyle=(0, (5, 10)))
            ax.add_collection(lc)

            plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
            plt.ylabel('Time in AF', fontsize=yaxisfontsize)
            plt.xlim([min_nu, max_nu])
            plt.ylim([0, 1.02])
            plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu),np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_label_coords(-0.13, 0.5)
            plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
            plt.tick_params(which='major', length=6)
            plt.tick_params(which='minor', length=3)
            plt.legend(fontsize=legendfontsize)
            plt.subplots_adjust(left=0.15)
            plt.savefig(figure_path + 'time_in_AF_CMP_vs_MF_vs_2MF.pdf', dpi=300)
            plt.close()

'''
########################################################################################################################
#                               Analysis of stable re-entrant circuits                                                 #
########################################################################################################################
stats_cCMP_filtered  = []
for i in np.arange(min_nu,max_nu + 0.5*step_nu, step_nu):
            current_sample = []
            for k in range(0, K, 1):
                    active_cells_per_step = np.load(common_path + "{:.2f}".format(i) + '\\CMP\\experiment_' + str(k) + '\\active_nodes_CMP_' + str(k) + '.npy')
                    moving_average        = pd.rolling_mean(active_cells_per_step, moving_average_period)
                    tag_cCMP              = mtd.check_functional(S, active_cells_per_step, moving_average, T, ma_threshold, segment_size, repeats)                    
                    if tag_cCMP == 0:
                        current_sample.append(np.load(common_path + "{:.2f}".format(i) +  '\\CMP\\experiment_' + str(k) + '\\af_risk_CMP_' + str(k) + '.npy').item())

            print('Stable re-entrant circuits detection procedure for nu: ' + str(i) + ' completed. Retained samples: ' + str(len(current_sample)))
            cCMP_avg_time_in_AF = np.average(current_sample)
            cCMP_std_time_in_AF = np.std(current_sample)
            cCMP_ub_time_in_AF  = cCMP_avg_time_in_AF + confidence_level * (cCMP_std_time_in_AF / float(np.sqrt(K)))
            cCMP_lb_time_in_AF  = cCMP_avg_time_in_AF - confidence_level * (cCMP_std_time_in_AF / float(np.sqrt(K)))
            stats_cCMP_filtered.append([cCMP_avg_time_in_AF, cCMP_lb_time_in_AF, cCMP_ub_time_in_AF, len(current_sample)])

cCMP_filtered_avg_time_in_AF_curve = np.array([stats_cCMP_filtered[i][0] for i in range(0, len(x_array),1)])
cCMP_filtered_ub_time_in_AF_curve  = np.array([stats_cCMP_filtered[i][1] for i in range(0, len(x_array),1)])
cCMP_filtered_lb_time_in_AF_curve  = np.array([stats_cCMP_filtered[i][2] for i in range(0, len(x_array),1)])
cCMP_filtered_excluded_ratio_curve = np.array([1.0 - stats_cCMP_filtered[i][3]/float(K) for i in range(0, len(x_array),1)])


fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
ax.plot(x_array, cCMP_filtered_avg_time_in_AF_curve, linewidth=1.5, marker='s', color=cCMP_color, markersize=4, label= 'cCMP')
ax.plot(x_array, cCMP_filtered_ub_time_in_AF_curve, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
ax.plot(x_array, cCMP_filtered_lb_time_in_AF_curve, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
segments = [[(x_array[ij], cCMP_filtered_ub_time_in_AF_curve[ij]), (x_array[ij], cCMP_filtered_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=cCMP_color, linewidths=1.5)
ax.add_collection(lc)

ax.plot(x_array, MF_avg_time_in_AF_curve, linewidth=1.5, marker='s', color= MF_color, markersize=4, label= 'MF')
ax.plot(x_array, MF_ub_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
ax.plot(x_array, MF_lb_time_in_AF_curve, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
segments = [[(x_array[ij], MF_ub_time_in_AF_curve[ij]), (x_array[ij], MF_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors= MF_color, linewidths=1.5)
ax.add_collection(lc)

#ax.plot(x_array, CMP_avg_time_in_AF_curve, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP')
#ax.plot(x_array, CMP_ub_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
#ax.plot(x_array, CMP_lb_time_in_AF_curve, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
#segments = [[(x_array[ij], CMP_ub_time_in_AF_curve[ij]), (x_array[ij], CMP_lb_time_in_AF_curve[ij])] for ij in range(0, len(x_array), 1)]
#lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
#ax.add_collection(lc)

#idx_1       = np.where(np.array(cCMP_filtered_avg_time_in_AF_curve) >= 0.5)[0][-1]
#idx_2       = np.where(np.array(cCMP_filtered_avg_time_in_AF_curve) < 0.5)[0][0]
#y_1         = cCMP_filtered_avg_time_in_AF_curve[idx_1]
#y_2         = cCMP_filtered_avg_time_in_AF_curve[idx_2]
#nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
#transitions = [[(nu_star, 0), (nu_star, 0.5)]]
#lc          = mc.LineCollection(transitions, colors=cCMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
#ax.add_collection(lc)

#idx_1       = np.where(np.array(CMP_avg_time_in_AF_curve) >= 0.5)[0][-1]
#idx_2       = np.where(np.array(CMP_avg_time_in_AF_curve) < 0.5)[0][0]
#y_1         = CMP_avg_time_in_AF_curve[idx_1]
#y_2         = CMP_avg_time_in_AF_curve[idx_2]
#nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
#transitions = [[(nu_star, 0), (nu_star, 0.5)]]
#lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
#ax.add_collection(lc)


ax.plot(x_array, cCMP_avg_time_in_AF_curve, linewidth=1.5, marker='s', color='red', markersize=4, linestyle=(0, (5, 10)))



plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
plt.ylabel('Time in AF', fontsize=yaxisfontsize)
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
ax2 = fig.add_axes([0.58, 0.35, 0.295, 0.35])
ax2.plot(x_array, cCMP_filtered_excluded_ratio_curve, marker='s', linewidth=1.5, markersize=4, color='red')
ax2.set_xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
ax2.set_xticklabels(np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
ax2.set_xlabel(r'$\nu_{\perp}$', fontsize = 14)
ax2.set_ylabel('Excluded Experiments', fontsize = 14)
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(which='both', width=1.5, labelsize=13)
ax2.tick_params(which='major', length=6)
ax2.tick_params(which='minor', length=3)
ax2.set_xlim([min_nu, max_nu])
ax2.set_ylim([0, 0.5])
ax2.set_yticklabels(np.arange(0.0,0.55, 0.1))
plt.savefig(figure_path + 'filtered_time_in_AF_cCMP_vs_CMP_vs_MF.pdf', dpi=300)
########################################################################################################################
#                               Analysis of stable re-entrant circuits                                                 #
########################################################################################################################
first_curve_cCMP     = []
first_curve_cCMP_ub  = []
first_curve_cCMP_lb  = []
second_curve_cCMP    = []
second_curve_cCMP_ub = []
second_curve_cCMP_lb = []
first_curve_CMP      = []
first_curve_CMP_ub   = []
first_curve_CMP_lb   = []
second_curve_CMP     = []
second_curve_CMP_ub  = []
second_curve_CMP_lb  = []
first_curve_MF       = []
first_curve_MF_ub    = []
first_curve_MF_lb    = []
second_curve_MF      = []
second_curve_MF_ub   = []
second_curve_MF_lb   = []
first_t_start     = 25000
first_t_end       = 75000
first_window      = first_t_end - first_t_start
second_t_start    = 950000
second_t_end      = 1000000
second_window     = second_t_end - second_t_start
for i in np.arange(min_nu,max_nu + 0.5*step_nu, step_nu):
            first_cCMP_sample  = []
            second_cCMP_sample = []
            first_CMP_sample   = []
            second_CMP_sample  = []
            first_MF_sample    = []
            second_MF_sample   = []
            for k in range(0, K, 1):
                    active_cells_per_step_cCMP    = np.load(common_path + "{:.2f}".format(i) + '\\cCMP\\experiment_' + str(k) + '\\active_nodes_cCMP_' + str(k) + '.npy')
                    active_cells_per_step_CMP     = np.load(common_path + "{:.2f}".format(i) + '\\CMP\\experiment_' + str(k) + '\\active_nodes_CMP_' + str(k) + '.npy')
                    active_structures_per_step_MF = np.load(common_path + "{:.2f}".format(i) + '\\MF\\experiment_' + str(k) + '\\active_particles_MF' + str(k) + '.npy')
                    first_cCMP_sample.append(np.sum(active_cells_per_step_cCMP[first_t_start:first_t_end] >= 220)/float(first_window))
                    second_cCMP_sample.append(np.sum(active_cells_per_step_cCMP[second_t_start:second_t_end] >= 220) / float(second_window))
                    first_CMP_sample.append(np.sum(active_cells_per_step_CMP[first_t_start:first_t_end] >= 220) / float(first_window))
                    second_CMP_sample.append(np.sum(active_cells_per_step_CMP[second_t_start:second_t_end] >= 220) / float(second_window))
                    first_MF_sample.append(np.sum(active_structures_per_step_MF[first_t_start:first_t_end] >= 1) / float(first_window))
                    second_MF_sample.append(np.sum(active_structures_per_step_MF[second_t_start:second_t_end] >= 1) / float(second_window))
                    print(k)

            first_cCMP_sample_avg = np.average(first_cCMP_sample)
            first_cCMP_sample_std = np.std(first_cCMP_sample)
            first_cCMP_sample_ub  = first_cCMP_sample_avg + confidence_level * (first_cCMP_sample_std / float(np.sqrt(K)))
            first_cCMP_sample_lb  = first_cCMP_sample_avg - confidence_level * (first_cCMP_sample_std / float(np.sqrt(K)))

            second_cCMP_sample_avg = np.average(second_cCMP_sample)
            second_cCMP_sample_std = np.std(second_cCMP_sample)
            second_cCMP_sample_ub  = second_cCMP_sample_avg + confidence_level * (second_cCMP_sample_std / float(np.sqrt(K)))
            second_cCMP_sample_lb  = second_cCMP_sample_avg - confidence_level * (second_cCMP_sample_std / float(np.sqrt(K)))

            first_CMP_sample_avg = np.average(first_CMP_sample)
            first_CMP_sample_std = np.std(first_CMP_sample)
            first_CMP_sample_ub  = first_CMP_sample_avg + confidence_level * (first_CMP_sample_std / float(np.sqrt(K)))
            first_CMP_sample_lb  = first_CMP_sample_avg - confidence_level * (first_CMP_sample_std / float(np.sqrt(K)))

            second_CMP_sample_avg = np.average(second_CMP_sample)
            second_CMP_sample_std = np.std(second_CMP_sample)
            second_CMP_sample_ub  = second_CMP_sample_avg + confidence_level * (second_CMP_sample_std / float(np.sqrt(K)))
            second_CMP_sample_lb  = second_CMP_sample_avg - confidence_level * (second_CMP_sample_std / float(np.sqrt(K)))

            first_MF_sample_avg = np.average(first_MF_sample)
            first_MF_sample_std = np.std(first_MF_sample)
            first_MF_sample_ub  = first_MF_sample_avg + confidence_level * (first_MF_sample_std / float(np.sqrt(K)))
            first_MF_sample_lb  = first_MF_sample_avg - confidence_level * (first_MF_sample_std / float(np.sqrt(K)))

            second_MF_sample_avg = np.average(second_MF_sample)
            second_MF_sample_std = np.std(second_MF_sample)
            second_MF_sample_ub  = second_MF_sample_avg + confidence_level * (second_MF_sample_std / float(np.sqrt(K)))
            second_MF_sample_lb  = second_MF_sample_avg - confidence_level * (second_MF_sample_std / float(np.sqrt(K)))



            first_curve_cCMP.append(first_cCMP_sample_avg)
            first_curve_cCMP_ub.append(first_cCMP_sample_ub)
            first_curve_cCMP_lb.append(first_cCMP_sample_lb)
            second_curve_cCMP.append(second_cCMP_sample_avg)
            second_curve_cCMP_ub.append(second_cCMP_sample_ub)
            second_curve_cCMP_lb.append(second_cCMP_sample_lb)

            first_curve_CMP.append(first_CMP_sample_avg)
            first_curve_CMP_ub.append(first_CMP_sample_ub)
            first_curve_CMP_lb.append(first_CMP_sample_lb)
            second_curve_CMP.append(second_CMP_sample_avg)
            second_curve_CMP_ub.append(second_CMP_sample_ub)
            second_curve_CMP_lb.append(second_CMP_sample_lb)

            first_curve_MF.append(first_MF_sample_avg)
            first_curve_MF_ub.append(first_MF_sample_ub)
            first_curve_MF_lb.append(first_MF_sample_lb)
            second_curve_MF.append(second_MF_sample_avg)
            second_curve_MF_ub.append(second_MF_sample_ub)
            second_curve_MF_lb.append(second_MF_sample_lb)



fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)
ax.plot(x_array, first_curve_cCMP, linewidth=1.5, marker='s', color=cCMP_color, markersize=4, label= 'cCMP')
ax.plot(x_array, first_curve_cCMP_ub, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
ax.plot(x_array, first_curve_cCMP_lb, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
segments = [[(x_array[ij], first_curve_cCMP_ub[ij]), (x_array[ij], first_curve_cCMP_lb[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=cCMP_color, linewidths=1.5)
ax.add_collection(lc)


ax.plot(x_array, first_curve_CMP, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP I')
ax.plot(x_array, first_curve_CMP_ub, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
ax.plot(x_array, first_curve_CMP_lb, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
segments = [[(x_array[ij], first_curve_CMP_ub[ij]), (x_array[ij], first_curve_CMP_lb[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=cCMP_color, linewidths=1.5)
ax.add_collection(lc)
if add_MF:
        ax.plot(x_array, first_curve_MF, linewidth=1.5, marker='s', color= MF_color, markersize=4, label= 'MF')
        ax.plot(x_array, first_curve_MF_ub, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
        ax.plot(x_array, first_curve_MF_lb, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
        segments = [[(x_array[ij], first_curve_MF_ub[ij]), (x_array[ij], first_curve_MF_lb[ij])] for ij in range(0, len(x_array), 1)]
        lc       = mc.LineCollection(segments, colors= MF_color, linewidths=1.5)
        ax.add_collection(lc)
idx_1       = np.where(np.array(first_curve_cCMP) >= 0.5)[0][-1]
idx_2       = np.where(np.array(first_curve_cCMP) < 0.5)[0][0]
y_1         = first_curve_cCMP[idx_1]
y_2         = first_curve_cCMP[idx_2]
nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
transitions = [[(nu_star, 0), (nu_star, 0.5)]]
lc          = mc.LineCollection(transitions, colors=cCMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
ax.add_collection(lc)

idx_1       = np.where(np.array(first_curve_CMP) >= 0.5)[0][-1]
idx_2       = np.where(np.array(first_curve_CMP) < 0.5)[0][0]
y_1         = first_curve_CMP[idx_1]
y_2         = first_curve_CMP[idx_2]
nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
transitions = [[(nu_star, 0), (nu_star, 0.5)]]
lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
ax.add_collection(lc)
if add_MF:
        idx_1       = np.where(np.array(first_curve_MF) >= 0.5)[0][-1]
        idx_2       = np.where(np.array(first_curve_MF) < 0.5)[0][0]
        y_1         = first_curve_MF[idx_1]
        y_2         = first_curve_MF[idx_2]
        nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
        transitions = [[(nu_star, 0), (nu_star, 0.5)]]
        lc          = mc.LineCollection(transitions, colors=MF_color, linewidths=1.25, linestyle=(0, (5, 10)))
        ax.add_collection(lc)
plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
plt.ylabel('Time in AF', fontsize=yaxisfontsize)
plt.xlim([min_nu, max_nu])
plt.ylim([0, 1.02])
plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu),np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_label_coords(-0.13, 0.5)
ax.text(0.0275, 1.125, '(b)', horizontalalignment='center', verticalalignment='center',fontsize=figuretagsize)
plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)
plt.legend(fontsize=legendfontsize)
plt.subplots_adjust(left=0.15)
plt.savefig(figure_path + 'first_interval_experiment.pdf', dpi=300)
plt.close()


fig, ax                   = plt.subplots(figsize=(6, 6), dpi=200)

ax.plot(x_array, second_curve_cCMP, linewidth=1.5, marker='s', color=cCMP_color, markersize=4, label= 'cCMP')
ax.plot(x_array, second_curve_cCMP_ub, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
ax.plot(x_array, second_curve_cCMP_lb, linewidth=0, marker='_', color=cCMP_color, markersize=4,label='')
segments = [[(x_array[ij], second_curve_cCMP_ub[ij]), (x_array[ij], second_curve_cCMP_lb[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=cCMP_color, linewidths=1.5)
ax.add_collection(lc)


ax.plot(x_array, second_curve_CMP, linewidth=1.5, marker='s', color=CMP_color, markersize=4, label= 'CMP II')
ax.plot(x_array, second_curve_CMP_ub, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
ax.plot(x_array, second_curve_CMP_lb, linewidth=0, marker='_', color=CMP_color, markersize=4,label='')
segments = [[(x_array[ij], second_curve_CMP_ub[ij]), (x_array[ij], second_curve_CMP_lb[ij])] for ij in range(0, len(x_array), 1)]
lc       = mc.LineCollection(segments, colors=CMP_color, linewidths=1.5)
ax.add_collection(lc)
if add_MF:
        ax.plot(x_array, second_curve_MF, linewidth=1.5, marker='s', color= MF_color, markersize=4, label= 'MF')
        ax.plot(x_array, second_curve_MF_ub, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
        ax.plot(x_array, second_curve_MF_lb, linewidth=0, marker='_', color= MF_color, markersize=4,label='')
        segments = [[(x_array[ij], second_curve_MF_ub[ij]), (x_array[ij], second_curve_MF_lb[ij])] for ij in range(0, len(x_array), 1)]
        lc       = mc.LineCollection(segments, colors= MF_color, linewidths=1.5)
        ax.add_collection(lc)
idx_1       = np.where(np.array(second_curve_cCMP) >= 0.5)[0][-1]
idx_2       = np.where(np.array(second_curve_cCMP) < 0.5)[0][0]
y_1         = second_curve_cCMP[idx_1]
y_2         = second_curve_cCMP[idx_2]
nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
transitions = [[(nu_star, 0), (nu_star, 0.5)]]
lc          = mc.LineCollection(transitions, colors=cCMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
ax.add_collection(lc)

idx_1       = np.where(np.array(second_curve_CMP) >= 0.5)[0][-1]
idx_2       = np.where(np.array(second_curve_CMP) < 0.5)[0][0]
y_1         = second_curve_CMP[idx_1]
y_2         = second_curve_CMP[idx_2]
nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
transitions = [[(nu_star, 0), (nu_star, 0.5)]]
lc          = mc.LineCollection(transitions, colors=CMP_color, linewidths=1.25, linestyle=(0, (5, 10)))
ax.add_collection(lc)

if add_MF:
        idx_1       = np.where(np.array(second_curve_MF) >= 0.5)[0][-1]
        idx_2       = np.where(np.array(second_curve_MF) < 0.5)[0][0]
        y_1         = second_curve_MF[idx_1]
        y_2         = second_curve_MF[idx_2]
        nu_star     = x_array[idx_2] - step_nu * ((0.5 - y_2) / float(y_1 - y_2))
        transitions = [[(nu_star, 0), (nu_star, 0.5)]]
        lc          = mc.LineCollection(transitions, colors=MF_color, linewidths=1.25, linestyle=(0, (5, 10)))
        ax.add_collection(lc)

plt.xlabel(r'$\nu_{\perp}$', fontsize=xaxisfontsize)
plt.ylabel('Time in AF', fontsize=yaxisfontsize)
plt.xlim([min_nu, max_nu])
plt.ylim([0, 1.02])
plt.xticks(np.arange(min_nu, max_nu + step_nu, 5 * step_nu),np.arange(min_nu, max_nu + step_nu, 5 * step_nu))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_label_coords(-0.13, 0.5)
ax.text(0.0275, 1.125, '(b)', horizontalalignment='center', verticalalignment='center',fontsize=figuretagsize)
plt.tick_params(which='both', width=1.5, labelsize=ticklabelsize)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)
plt.legend(fontsize=legendfontsize)
plt.subplots_adjust(left=0.15)
plt.savefig(figure_path + 'MF_experiment.pdf', dpi=300)
plt.close()
'''

