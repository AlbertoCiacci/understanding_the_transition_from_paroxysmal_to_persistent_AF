######################################################
#               Import Libraries                     #
######################################################
import numpy as np                   # Import numpy for a vector approach in the coding
from time import clock               # From time we import clock in order to track the execution time of the algorithm
import multiprocessing as mp         # Multiprocessing allows us for parallel computing
import os,sys                        # Import the os library in order to operate with directories
import random as rdn                 # Random library to create seeds
import itertools as it               # Import itertools to merge list of lists
num_cores      = mp.cpu_count()                  # Retrieve the number of cores of the current machine
project_subdir = os.getcwd()                     # Get the current subdirectory of the project
project_dir    = os.path.dirname(project_subdir) # Get the mother directory of the project

######################################################
#               Import Classes                       #
######################################################
from models_video import CMP_models                        # Import the class that contains the models investigated in this project (cellular automata)
######################################################
#               Import Functions                     #
######################################################
#import CMP_Methods_New as sm                                  # Import support functions from the dedicated scripts

######################################################
#                 General Settings                   #
######################################################
L                  = 200                              # Grid size
nu                 = 0.10                             # nu
S                  = 1e6                              # Steps per simulation [CMP]
T                  = 220                              # Pacing time
tau                = 50                               # Refractory time
delta              = 0.01                             # Dysfunctional ratio
epsilon            = 0.05                             # Failure probability
######################################################
#          Model Parameters                          #
######################################################
target_simulations = [51]#[143]
output_folder      = "D:\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\videos"
cCMP_folder        = "D:\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\risk analysis\\" + "{:.2f}".format(nu)+ "\\cCMP\\"
CMP_folder         = "D:\\Dropbox\\Documents\\Academic\\PhD\\Research\\Imperial College London\\AF transition\\risk analysis\\" + "{:.2f}".format(nu)+ "\\CMP\\"
os.makedirs(output_folder, exist_ok=True)
######################################################
#          Graphic parameters                        #
######################################################
starting_capture_time = 0#201900#220400       # Set the initial capture time
ending_capture_time   = 1000#201950#220600     # Set the final capture time
min_x                 = 0
max_x                 = 200
min_y                 = 0
max_y                 = 200
produce_videos_1      = 0
produce_schemes_1     = 0
produce_videos_2      = 1
produce_schemes_2     = 0
fps                   = 5
interval              = 100
target_frames         = [197510,197521,197572,197591,197624]
######################################################
#                  Algorithm                         #
######################################################
# Activate the clock to measure the time taken by the whole routine to complete the process
begin        = clock()
model        = CMP_models(L, S, T, tau, delta, epsilon, nu,starting_capture_time, ending_capture_time, min_x, max_x, min_y, max_y, produce_videos_1, produce_schemes_1, produce_videos_2, produce_schemes_2, fps, interval)
temp_outputs = []
target_f     = model.cmp_simulator
for sim_id in target_simulations:
    model.cmp_simulator(output_folder, target_frames, cCMP_folder, CMP_folder, sim_id)
end = clock()
print('The entire process has run in ' + str(end-begin) + ' seconds')