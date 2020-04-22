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
from MF_models import MF_models                        # Import the class that contains the models investigated in this project (cellular automata)

######################################################
#                 General Settings                   #
######################################################
used_cores          = num_cores - 1                    # The number of cores of the processor that are used for parallelized computations
K                   = 200                              # Number of trials (Monte Carlo)
L                   = 200                              # Grid size
S                   = 1e6                              # Steps per simulation [CMP]
T                   = 220                              # Pacing time
tau                 = 50                               # Refractory time
delta               = 0.01                             # Dysfunctional ratio
epsilon             = 0.05                             # Failure probability
scaling_factor      = 2.0                              # Multiplier for the scaled version of the MF model
seed_type           = 1                                # 1 for new seeds, 2 for replication
######################################################
#          Model Parameters                          #
######################################################
min_nu              = 0.05                             # Minimum nu
max_nu              = 0.20                             # Maximum nu
step_nu             = 0.01                             # Step size between consecutives nu
######################################################
#            Output [results] folder                 #
######################################################

common_path           = "D:\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\risk analysis"
seeds_path            = common_path + "\\MF seeds\\"
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
    models      = [MF_models(L, S, T, epsilon, scaling_factor) for nu in np.arange(min_nu, max_nu+step_nu, step_nu)]
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
            target_f =  m.mf_simulator
            # Reset the completed trials counter
            completed_trials    = 0
            # Perform a Monte Carlo simulation until the results of K trials have been collected
            while completed_trials < K:
                        # Create seeds
                        seeds = rdn.sample(range(int(2 ** 32 - 1)), used_cores)
                        print('Seeds for nu: ' + "{:.2f}".format(nu) + ', simulations ids [' + str(completed_trials ) +','+str(completed_trials + used_cores ) +'] created')
                        # CS Lengths
                        cs_lengths           = [np.load(common_path + "\\" + "{:.2f}".format(nu) + "\\cCMP\\experiment_" + str(completed_trials + ij) + "\\cs_length_cCMP_" + str(completed_trials + ij) + ".npy") for ij in range(0, used_cores, 1)]
                        # Outputs array
                        temp_outputs         = []
                        # Open pools
                        pool                 =  mp.Pool(processes=used_cores)
                        # Perform CA simulations in parallel across the user defined number of cores
                        temp_outputs.extend([single_output.get() for single_output in [pool.apply_async(target_f, args=(seeds[ix], cs_lengths[ix], nu, completed_trials + ix ,)) for ix in range(0, used_cores,1)]])
                        # Close and join the pools
                        pool.close()
                        pool.join()
                        # Save the results
                        for i in range(0, len(temp_outputs), 1):
                                                subfolder = common_path + "\\" + "{:.2f}".format(nu) + "\\MF\\experiment_" + str(completed_trials + i) + "\\"
                                                os.makedirs(subfolder, exist_ok=True)
                                                np.save(subfolder + 'af_risk_MF_' + str(completed_trials + i) + '.npy', temp_outputs[i][0][0])
                                                np.save(subfolder + 'af_flag_MF_' + str(completed_trials + i) + '.npy', temp_outputs[i][0][1])
                                                np.save(subfolder + 'active_particles_MF' + str(completed_trials + i) + '.npy', temp_outputs[i][0][2])

                                                subfolder = common_path + "\\" + "{:.2f}".format(nu) + "\\eMF\\experiment_" + str(completed_trials + i) + "\\"
                                                os.makedirs(subfolder, exist_ok=True)
                                                np.save(subfolder + 'af_risk_eMF_' + str(completed_trials + i) + '.npy',temp_outputs[i][1][0])
                                                np.save(subfolder + 'af_flag_eMF_' + str(completed_trials + i) + '.npy',temp_outputs[i][1][1])
                                                np.save(subfolder + 'active_particles_eMF' + str(completed_trials + i) + '.npy', temp_outputs[i][1][2])

                                                subfolder = common_path + "\\" + "{:.2f}".format(nu) + "\\cMF\\experiment_" + str(completed_trials + i) + "\\"
                                                os.makedirs(subfolder, exist_ok=True)
                                                np.save(subfolder + 'af_risk_cMF_' + str(completed_trials + i) + '.npy', temp_outputs[i][2][0])

                                                subfolder = common_path + "\\" + "{:.2f}".format(nu) + "\\2MF\\experiment_" + str(completed_trials + i) + "\\"
                                                os.makedirs(subfolder, exist_ok=True)
                                                np.save(subfolder + 'af_risk_2MF_' + str(completed_trials + i) + '.npy',temp_outputs[i][3][0])
                                                np.save(subfolder + 'af_flag_2MF_' + str(completed_trials + i) + '.npy',temp_outputs[i][3][1])
                                                np.save(subfolder + 'active_particles_2MF' + str(completed_trials + i) + '.npy', temp_outputs[i][3][2])


                        # Update the variable that counts the number of completed Monte Carlo trials
                        completed_trials += used_cores
            # Update nu
            nu_counter += 1


    # Terminate the clock to measure the time taken by the whole routine to complete the process
    end = clock()
    # Print the time required to run the entire routine
    print('The entire process has run in ' + str(end-begin) + ' seconds')