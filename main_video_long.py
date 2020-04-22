######################################################
#               Import Libraries                     #
######################################################
import numpy as np                   # Import numpy for a vector approach in the coding
from time import clock               # From time we import clock in order to track the execution time of the algorithm
import os,sys                        # Import the os library in order to operate with directories
import random as rdn                 # Random library to create seeds
import itertools as it               # Import itertools to merge list of lists
project_subdir = os.getcwd()                     # Get the current subdirectory of the project
project_dir    = os.path.dirname(project_subdir) # Get the mother directory of the project

######################################################
#               Import Classes                       #
######################################################
from models_video_long import CMP_models                        # Import the class that contains the models investigated in this project (cellular automata)
######################################################
#               Import Functions                     #
######################################################
#import CMP_Methods_New as sm                                  # Import support functions from the dedicated scripts

######################################################
#                 General Settings                   #
######################################################
L                  = 200                              # Grid size
nu                 = 0.05                             # nu
S                  = 1e6                              # Steps per simulation [CMP]
T                  = 220                              # Pacing time
tau                = 50                               # Refractory time
delta              = 0.01                             # Dysfunctional ratio
epsilon            = 0.05                             # Failure probability
######################################################
#          Model Parameters                          #
######################################################
target_simulations = [86]#[143]
output_folder      = "D:\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\long experiments\\extensions (1e9)"
os.makedirs(output_folder, exist_ok=True)
######################################################
#          Graphic parameters                        #
######################################################
starting_capture_time = 200000#41300#220400       # Set the initial capture time
ending_capture_time   = 200500#41500#220600     # Set the final capture time
min_x                 = 0#10
max_x                 = 80
min_y                 = 164
max_y                 = 168
produce_videos        = 0
produce_schemes       = 1
fps                   = 5
interval              = 100
target_frames         = [197510,197521,197572,197591,197624]
######################################################
#                  Algorithm                         #
######################################################
# Activate the clock to measure the time taken by the whole routine to complete the process
begin        = clock()
model        = CMP_models(L, S, T, tau, delta, epsilon, nu,starting_capture_time, ending_capture_time, min_x, max_x, min_y, max_y, produce_videos, produce_schemes, fps, interval)
for sim_id in target_simulations:
    seed      = np.load(output_folder + "\\seeds\\" + "{:.2f}".format(nu)+ "\\" + str(sim_id) + ".npy")
    target_ts = np.load(output_folder + "\\" + "{:.2f}".format(nu)+ "\\CMP\\experiment_" + str(sim_id) + "\\active_nodes_CMP_" + str(sim_id) + ".npy")
    model.cmp_simulator(seed, target_ts, output_folder, target_frames, sim_id)
end = clock()
print('The entire process has run in ' + str(end-begin) + ' seconds')