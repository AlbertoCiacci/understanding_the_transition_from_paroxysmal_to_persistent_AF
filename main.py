######################################################
#               Import Libraries                     #
######################################################
import numpy as np                   # Import numpy for a vector approach in the coding
from time import clock               # From time we import clock in order to track the execution time of the algorithm
import multiprocessing as mp         # Multiprocessing allows us for parallel computing
import os,sys                        # Import the os library in order to operate with directories
import random as rdn                 # Random library to create seeds
num_cores      = mp.cpu_count()                  # Retrieve the number of cores of the current machine
project_subdir = os.getcwd()                     # Get the current subdirectory of the project
project_dir    = os.path.dirname(project_subdir) # Get the mother directory of the project

######################################################
#               Import Classes                       #
######################################################
from models import CMP_models                        # Import the class that contains the models investigated in this project (cellular automata)

######################################################
#                 General Settings                   #
######################################################
used_cores          = num_cores - 3                    # The number of cores of the processor that are used for parallelized computations
K                   = 200                              # Number of trials (Monte Carlo)
L                   = 200                              # Grid size
S                   = 1e5                              # Steps per simulation [CMP]
T                   = 220                              # Pacing time
tau                 = 50                               # Refractory time
delta               = 0.0005                             # Dysfunctional ratio
epsilon             = 0.05                            # Failure probability
t_af                = L + 20                           # Minimum number of excited cells for considering the system in AF (approach 1)
seed_type           = 1                                # 1 for new seeds, 2 for replication
analysis_type       = 2                                # 1 for risk, 2 for induction
######################################################
#          Model Parameters                          #
######################################################
min_nu              = 0.05                             # Minimum nu
max_nu              = 0.20                             # Maximum nu
step_nu             = 0.01                             # Step size between consecutives nu
saving_idx_shift    = 0                                # Shift in the index of each experiment
######################################################
#            Output [results] folder                 #
######################################################

if analysis_type == 1:
    common_path   = "C:\\Users\\Alberto Ciacci\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\risk analysis"
else:
    common_path   = "C:\\Users\\Alberto Ciacci\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\induction analysis short\\"+str(delta)
seeds_path            = common_path + "\\seeds\\"
os.makedirs(seeds_path, exist_ok=True)
os.makedirs(common_path, exist_ok=True)


######################################################
#                  Algorithm                         #
######################################################

# Enter a parallel computing environment
if __name__ == '__main__':

    # Activate the clock to measure the time taken by the whole routine to complete the process
    begin = clock()

    ######################################################
    #                   Grid Models                      #
    ######################################################

    # Create an array of instances for the cython class 'CMP_Models' for each transversal ratio in [nu_min,nu_max].
    models      = [CMP_models(L, S, T, tau, delta, epsilon, nu, t_af) for nu in np.arange(min_nu, max_nu+0.5*step_nu, step_nu)]
    # Set a counter to track the current nu
    nu_counter  = 0
    # Loop over the class instances
    for m in models:
            # Find the current nu
            nu = min_nu + nu_counter * step_nu
            # Create the seed folder
            seedfolder = seeds_path + "{:.2f}".format(nu) + "\\"
            os.makedirs(seedfolder, exist_ok=True)
            # Determine the target function
            if analysis_type == 1:
                    target_f =  m.cmp_simulator
            else:
                    target_f =  m.cmp_simulator_induction
            # Reset the completed trials counter
            completed_trials    = 0
            # Perform a Monte Carlo simulation until the results of K trials have been collected
            while completed_trials < K:
                        # Create seeds
                        seeds = rdn.sample(range(int(2 ** 32 - 1)), used_cores)
                        print('Seeds for nu: ' + "{:.2f}".format(nu) + ', simulations ids [' + str(completed_trials + saving_idx_shift ) +','+str(completed_trials + used_cores + saving_idx_shift ) +'] created')
                        # Outputs array
                        temp_outputs         = []
                        # Open pools
                        pool                 =  mp.Pool(processes=used_cores)
                        # Perform CA simulations in parallel across the user defined number of cores
                        temp_outputs.extend([single_output.get() for single_output in [pool.apply_async(target_f, args=(seeds[ix],completed_trials + ix + saving_idx_shift ,)) for ix in range(0, used_cores,1)]])
                        # Close and join the pools
                        pool.close()
                        pool.join()
                        # Save the results
                        for i in range(0, len(temp_outputs), 1):
                                                if analysis_type == 1:
                                                        subfolder = common_path + "\\" + "{:.2f}".format(nu) + "\\cCMP\\experiment_" + str(completed_trials + i + saving_idx_shift) + "\\"
                                                        os.makedirs(subfolder, exist_ok=True)
                                                        np.save(subfolder + 'down_links_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][0][0])
                                                        np.save(subfolder + 'af_risk_cCMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][0][1])
                                                        np.save(subfolder + 'af_flag_cCMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][0][2])
                                                        np.save(subfolder + 'active_nodes_cCMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][0][3])
                                                        np.save(subfolder + 'dysfunctional_cCMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][0][4])
                                                        np.save(subfolder + 'cs_length_cCMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][0][5])
                                                        np.save(subfolder + 'cs_paths_cCMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][0][6])
                                                        np.savez_compressed(subfolder + 'dysf_matrix_cCMP_' + str(completed_trials + i + saving_idx_shift) + '.npz',temp_outputs[i][0][7])

                                                        subfolder = common_path + "\\" + "{:.2f}".format(nu) + "\\CMP\\experiment_" + str(completed_trials + i + saving_idx_shift) + "\\"
                                                        os.makedirs(subfolder, exist_ok=True)
                                                        np.save(subfolder + 'down_links_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][1][0])
                                                        np.save(subfolder + 'af_risk_CMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][1][1])
                                                        np.save(subfolder + 'af_flag_CMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][1][2])
                                                        np.save(subfolder + 'active_nodes_CMP_' + str(completed_trials + i + saving_idx_shift) + '.npy',temp_outputs[i][1][3])
                                                        np.save(subfolder + 'dysfunctional_CMP_' + str(completed_trials + i + saving_idx_shift) + '.npy',temp_outputs[i][1][4])
                                                        np.save(subfolder + 'cs_length_CMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][1][5])
                                                        np.save(subfolder + 'cs_paths_CMP_' + str(completed_trials + i + saving_idx_shift) + '.npy', temp_outputs[i][1][6])
                                                        np.savez_compressed(subfolder + 'dysf_matrix_CMP_' + str(completed_trials + i + saving_idx_shift) + '.npz', temp_outputs[i][1][7])

                                                else:
                                                    subfolder = common_path + "\\" + "{:.2f}".format(nu) + "\\experiment_" + str(completed_trials + i + saving_idx_shift) + "\\"
                                                    os.makedirs(subfolder, exist_ok=True)
                                                    np.save(subfolder + 'AF_flag_CCMP_' + str(completed_trials + i + saving_idx_shift) + '.npy',temp_outputs[i][0])
                                                    np.save(subfolder + 'AF_flag_CMP_' + str(completed_trials + i + saving_idx_shift) + '.npy',temp_outputs[i][1])
                                                    np.save(subfolder + 'AF_flag_MF_' + str(completed_trials + i + saving_idx_shift) + '.npy',temp_outputs[i][2])






                        # Update the variable that counts the number of completed Monte Carlo trials
                        completed_trials += used_cores
            # Update nu
            nu_counter += 1


    # Terminate the clock to measure the time taken by the whole routine to complete the process
    end = clock()
    # Print the time required to run the entire routine
    print('The entire process has run in ' + str(end-begin) + ' seconds')