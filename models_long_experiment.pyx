######################################################################
#                                                                    #
#                     Import Libraries                               #
#                                                                    #
######################################################################
import time
import numpy as np
######################################################################
#                                                                    #
#                     Import Scripts                                 #
#                                                                    #
######################################################################
import methods as mtd

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport cython
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
TYPE1 = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t   TYPE1_t


cdef class CMP_models:
    cdef int L, S, T, tau, t_af
    cdef double delta, epsilon, nu

    def __init__(self, L, S, T, tau, delta, epsilon, nu, t_af):
            # This is equivalent to a C++ constructor

            self.L                    = L                       # Grid size
            self.S                    = S                       # Number of steps
            self.T                    = T                       # Pacemaker period
            self.tau                  = tau                     # Refractory period
            self.delta                = delta                   # Dysfunctional cell ratio
            self.epsilon              = epsilon                 # Failure rate for dysfunctional cells
            self.nu                   = nu                      # Transversal connections rate
            self.t_af                 = t_af                  # AF threshold for type 1 calculation


    # bound checks and checks for negative indexes are turned off in order
    # to minimize the execution time. Binding and linetrace are activated
    # in order to profile the function and measure the timing of each line
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.linetrace(True)
    @cython.binding(True)
    # "def" can type its arguments but not have a return type. The type of the
    # arguments for a "def" function is checked at run-time when entering the
    # function.
    cpdef cmp_simulator(self, seed, simulation_id):

        # Set the seed
        random_generator = np.random.RandomState(seed)
        ################################################################################################################
        #                                                                                                              #
        #                                                Phase I                                                       #
        #                                                                                                              #
        ################################################################################################################
        cdef int n
        # the numpy array inputs are turned into 'C' array objects and conveniently renamed
        # grid represents the flattened LxL grid
        cdef np.ndarray grid          = np.zeros((self.L*self.L),dtype= TYPE1)
        # Track the primary critical structure
        dysfunctional,  down_links, cs_length_ccmp, cs_paths_ccmp, candidate_paths, delta_vector = mtd.build_grid_ccmp_new_rg(random_generator, self.nu,self.L,self.tau, self.delta)
        # Find the up links
        up_links                      = mtd.shift2(down_links, self.L)
        # Calculate the number of dysfunctional cells
        cdef int n_dysf               = np.count_nonzero(dysfunctional)
        # Construct a (1 x n_dysf) array containing the indexes of the dysfunctional cells
        cdef dysf_loc                 = np.array(np.where(dysfunctional == 1)[0],dtype= TYPE1)
        # Store indicators for healthy cells
        cdef np.ndarray healthy_cells = np.ones((self.L*self.L), dtype= np.bool)
        # left_links and right_links are boolean arrays telling whether a cell has a right/left connection
        cdef np.ndarray left_links    = np.ones((self.L*self.L), dtype= np.bool)
        cdef np.ndarray right_links   = np.ones((self.L*self.L), dtype= np.bool)
        # left_edge and right_edges are integer arrays containing the indexes of the cells lying in the edges
        cdef np.ndarray left_edge     = np.arange(0,self.L*self.L,self.L,dtype = TYPE1)
        cdef np.ndarray right_edge    = np.arange(self.L-1,self.L*self.L,self.L,dtype = TYPE1)
        cdef np.ndarray top_edge      = np.arange(0,self.L,1,dtype = TYPE1)
        cdef np.ndarray bottom_edge   = np.arange(self.L*self.L - self.L,self.L*self.L,1,dtype = TYPE1)
        # set the left links indicators on the left edge and the right links indicators on the right edge to FALSE
        left_links[left_edge]         = False
        right_links[right_edge]       = False
        # Track the number of active cells per step
        cdef np.ndarray active_log    = np.array(np.zeros(self.S),dtype=TYPE1)
        # Set the pacemaker type
        pacemaker_cells               = left_edge
        pacemaker_bool                = np.zeros((self.L*self.L), dtype= np.bool)
        # Create the dysfunctional matrix
        #dysf_matrix_ccmp              = random_generator.uniform(0.0, 1.0, (self.S, n_dysf)) > self.epsilon
        # Set the active cells array
        active                        = (grid == self.tau + 1)
        # Initiate the timer
        begin                         = time.clock()
        # Loop over the S steps, starting from step number 2 as the first one simply consists in the left edge pacemaker
        '''
        for n in range(0,self.S,1):
                            grid -= 1

                            right_neighbors = mtd.shift2((active*right_links), 1)
                            left_neighbors  = mtd.shift2((active*left_links), -1)
                            down_neighbors  = mtd.shift2((active*down_links), self.L)
                            up_neighbors    = mtd.shift2((active*up_links), -self.L)

                            resting_cells   = grid < 1
                            potential_sites = (right_neighbors) | (left_neighbors)| (down_neighbors) | (up_neighbors)

                            temp_healthy            = np.copy(healthy_cells)
                            temp_healthy[dysf_loc]  = (random_generator.uniform(0.0, 1.0, n_dysf) > self.epsilon)
                            #temp_healthy[dysf_loc]  = dysf_matrix_ccmp[n,:]


                            #grid -= 1

                            temp_pacemaker_bool = np.copy(pacemaker_bool)
                            temp_pacemaker_bool[pacemaker_cells] = (n % self.T == 0)*True

                            active                = ((resting_cells)*(potential_sites)*(temp_healthy)) | ((temp_pacemaker_bool)*(resting_cells))
                            grid[active]          = self.tau + 1

                            active_log[n]         = np.count_nonzero(active)



        ############################################################
        #                Calculate AF, approach 1                  #
        ############################################################
        AF_boolean         = active_log >= self.t_af
        durations          = np.array(mtd.cmp_durations(AF_boolean))
        af_flag_ccmp       = (np.count_nonzero(durations >= self.t_af) > 0)
        af_risk_ccmp       = np.sum(AF_boolean)/float(self.S)
        active_log_ccmp    = active_log
        dysfunctional_ccmp = dysfunctional
        # Terminate the timer
        '''
        end      = time.clock()
        # Prepare the output
        #print('Controlled CMP: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(self.nu, 2)) + ', Time in AF: ' + str(af_risk_ccmp))


        ################################################################################################################
        #                                                                                                              #
        #                                                Phase II                                                      #
        #                                                                                                              #
        ################################################################################################################

        # grid represents the flattened LxL grid
        grid                          = np.zeros((self.L*self.L),dtype= TYPE1)
        # Add dysfunctional cells outside the structures
        dysfunctional_cmp             = np.zeros((self.L*self.L),dtype=np.bool)
        dysf_mask                     = (random_generator.uniform(0,1,self.L*self.L) <= delta_vector)
        dysfunctional_cmp[dysf_mask]  = True
        # Find the cmp structures
        cs_length_cmp, cs_paths_cmp   = mtd.check_structures_cmp_new(dysfunctional_cmp, candidate_paths, self.tau)
        # Calculate the number of dysfunctional cells
        n_dysf                        = np.count_nonzero(dysfunctional_cmp)
        # Construct a (1 x n_dysf) array containing the indexes of the dysfunctional cells
        dysf_loc                      = np.array(np.where(dysfunctional_cmp == 1)[0],dtype= TYPE1)
        # Store indicators for healthy cells
        healthy_cells                 = np.ones((self.L*self.L), dtype= np.bool)
        # left_links and right_links are boolean arrays telling whether a cell has a right/left connection
        left_links                    = np.ones((self.L*self.L), dtype= np.bool)
        right_links                   = np.ones((self.L*self.L), dtype= np.bool)
        # left_edge and right_edges are integer arrays containing the indexes of the cells lying in the edges
        left_edge                     = np.arange(0,self.L*self.L,self.L,dtype = TYPE1)
        right_edge                    = np.arange(self.L-1,self.L*self.L,self.L,dtype = TYPE1)
        top_edge                      = np.arange(0,self.L,1,dtype = TYPE1)
        bottom_edge                   = np.arange(self.L*self.L - self.L,self.L*self.L,1,dtype = TYPE1)
        # set the left links indicators on the left edge and the right links indicators on the right edge to FALSE
        left_links[left_edge]         = False
        right_links[right_edge]       = False
        # Track the number of active cells per step
        active_log                    = np.array(np.zeros(self.S),dtype=TYPE1)
        # Set the pacemaker type
        pacemaker_cells               = left_edge
        pacemaker_bool                = np.zeros((self.L*self.L), dtype= np.bool)
        # Create the dysfunctional matrix
        #dysf_matrix_cmp               = random_generator.uniform(0.0, 1.0, (self.S, n_dysf)) > self.epsilon
        # Set the active cells array
        active                        = (grid == self.tau + 1)
        # Initiate the timer
        begin                         = time.clock()
        # Loop over the S steps, starting from step number 2 as the first one simply consists in the left edge pacemaker
        for n in range(0,self.S,1):
                        grid -= 1

                        right_neighbors = mtd.shift2((active*right_links), 1)
                        left_neighbors  = mtd.shift2((active*left_links), -1)
                        down_neighbors  = mtd.shift2((active*down_links), self.L)
                        up_neighbors    = mtd.shift2((active*up_links), -self.L)

                        resting_cells   = grid < 1
                        potential_sites = (right_neighbors) | (left_neighbors)| (down_neighbors) | (up_neighbors)

                        temp_healthy              = np.copy(healthy_cells)
                        temp_healthy[dysf_loc]    = (random_generator.uniform(0.0, 1.0, n_dysf) > self.epsilon)
                        #temp_healthy[dysf_loc]   = dysf_matrix_cmp[n,:]


                        #grid -= 1


                        temp_pacemaker_bool = np.copy(pacemaker_bool)
                        temp_pacemaker_bool[pacemaker_cells] = (n % self.T == 0)*True

                        active                = ((resting_cells)*(potential_sites)*(temp_healthy)) | ((temp_pacemaker_bool)*(resting_cells))
                        grid[active]          = self.tau + 1

                        active_log[n]         = np.count_nonzero(active)


                        if n == 2000000:
                            import matplotlib.pyplot as plt
                            import pandas as pd
                            fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                            ax.plot(active_log[0:n], linewidth=0.05, color='k', label=r'$a(t)$')
                            ax.plot(pd.rolling_mean(active_log[0:n],220), linewidth=1.0, color='red',label=r'$\langle a(t) \rangle$')
                            ax.set_xlabel('$t$', fontsize=18)
                            ax.set_ylabel(r'$a(t)$', fontsize=18)
                            ax.set_xlim([0, self.S])
                            ax.set_ylim([0, 5 * self.L + 1])
                            ax.set_yticks(range(0, 5 * self.L + 1, int(0.5 * self.L)))
                            ax.set_yticklabels(range(0, 5 * self.L + 1, int(0.5 * self.L)))
                            plt.tick_params(which='both', width=1.5, labelsize=18)
                            plt.tick_params(which='major', length=6)
                            plt.tick_params(which='minor', length=3)
                            ax.yaxis.set_label_coords(-0.18, 0.5)
                            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                            plt.rc('font', **{'size': str(18)})
                            plt.subplots_adjust(left=0.20)
                            plt.show()



        ############################################################
        #                Calculate AF, approach 1                  #
        ############################################################
        AF_boolean         = active_log >= self.t_af
        durations          = np.array(mtd.cmp_durations(AF_boolean))
        af_flag_cmp        = (np.count_nonzero(durations >= self.t_af) > 0)
        af_risk_cmp        = np.sum(AF_boolean)/float(self.S)
        active_log_cmp     = active_log

        # Terminate the timer
        end      = time.clock()
        # Prepare the output
        print('Standard CMP: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(self.nu, 2)) + ', Time in AF: ' + str(af_risk_cmp))

        return [down_links, af_risk_cmp, af_flag_cmp, active_log_cmp, dysfunctional_cmp, cs_length_cmp, cs_paths_cmp]


    cpdef cmp_simulator_replication(self, target_ts, seed, simulation_id):

        # Set the seed
        random_generator = np.random.RandomState(seed)
        ################################################################################################################
        #                                                                                                              #
        #                                                Phase I                                                       #
        #                                                                                                              #
        ################################################################################################################
        cdef int n
        # the numpy array inputs are turned into 'C' array objects and conveniently renamed
        # grid represents the flattened LxL grid
        cdef np.ndarray grid          = np.zeros((self.L*self.L),dtype= TYPE1)
        # Track the primary critical structure
        dysfunctional,  down_links, cs_length_ccmp, cs_paths_ccmp, candidate_paths, delta_vector = mtd.build_grid_ccmp_new_rg(random_generator, self.nu,self.L,self.tau, self.delta)
        # Find the up links
        up_links                      = mtd.shift2(down_links, self.L)
        # Calculate the number of dysfunctional cells
        cdef int n_dysf               = np.count_nonzero(dysfunctional)
        # Construct a (1 x n_dysf) array containing the indexes of the dysfunctional cells
        cdef dysf_loc                 = np.array(np.where(dysfunctional == 1)[0],dtype= TYPE1)
        # Store indicators for healthy cells
        cdef np.ndarray healthy_cells = np.ones((self.L*self.L), dtype= np.bool)
        # left_links and right_links are boolean arrays telling whether a cell has a right/left connection
        cdef np.ndarray left_links    = np.ones((self.L*self.L), dtype= np.bool)
        cdef np.ndarray right_links   = np.ones((self.L*self.L), dtype= np.bool)
        # left_edge and right_edges are integer arrays containing the indexes of the cells lying in the edges
        cdef np.ndarray left_edge     = np.arange(0,self.L*self.L,self.L,dtype = TYPE1)
        cdef np.ndarray right_edge    = np.arange(self.L-1,self.L*self.L,self.L,dtype = TYPE1)
        cdef np.ndarray top_edge      = np.arange(0,self.L,1,dtype = TYPE1)
        cdef np.ndarray bottom_edge   = np.arange(self.L*self.L - self.L,self.L*self.L,1,dtype = TYPE1)
        # set the left links indicators on the left edge and the right links indicators on the right edge to FALSE
        left_links[left_edge]         = False
        right_links[right_edge]       = False
        # Track the number of active cells per step
        cdef np.ndarray active_log    = np.array(np.zeros(self.S),dtype=TYPE1)
        # Set the pacemaker type
        pacemaker_cells               = left_edge
        pacemaker_bool                = np.zeros((self.L*self.L), dtype= np.bool)
        # Create the dysfunctional matrix
        #dysf_matrix_ccmp              = random_generator.uniform(0.0, 1.0, (self.S, n_dysf)) > self.epsilon
        # Set the active cells array
        active                        = (grid == self.tau + 1)
        # Initiate the timer
        begin                         = time.clock()
        # Loop over the S steps, starting from step number 2 as the first one simply consists in the left edge pacemaker
        '''
        for n in range(0,self.S,1):
                            grid -= 1

                            right_neighbors = mtd.shift2((active*right_links), 1)
                            left_neighbors  = mtd.shift2((active*left_links), -1)
                            down_neighbors  = mtd.shift2((active*down_links), self.L)
                            up_neighbors    = mtd.shift2((active*up_links), -self.L)

                            resting_cells   = grid < 1
                            potential_sites = (right_neighbors) | (left_neighbors)| (down_neighbors) | (up_neighbors)

                            temp_healthy            = np.copy(healthy_cells)
                            temp_healthy[dysf_loc]  = (random_generator.uniform(0.0, 1.0, n_dysf) > self.epsilon)
                            #temp_healthy[dysf_loc]  = dysf_matrix_ccmp[n,:]


                            #grid -= 1

                            temp_pacemaker_bool = np.copy(pacemaker_bool)
                            temp_pacemaker_bool[pacemaker_cells] = (n % self.T == 0)*True

                            active                = ((resting_cells)*(potential_sites)*(temp_healthy)) | ((temp_pacemaker_bool)*(resting_cells))
                            grid[active]          = self.tau + 1

                            active_log[n]         = np.count_nonzero(active)



        ############################################################
        #                Calculate AF, approach 1                  #
        ############################################################
        AF_boolean         = active_log >= self.t_af
        durations          = np.array(mtd.cmp_durations(AF_boolean))
        af_flag_ccmp       = (np.count_nonzero(durations >= self.t_af) > 0)
        af_risk_ccmp       = np.sum(AF_boolean)/float(self.S)
        active_log_ccmp    = active_log
        dysfunctional_ccmp = dysfunctional
        # Terminate the timer
        '''
        end      = time.clock()
        # Prepare the output

        #print('Controlled CMP: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(self.nu, 2)) + ', Time in AF: ' + str(af_risk_ccmp))


        ################################################################################################################
        #                                                                                                              #
        #                                                Phase II                                                      #
        #                                                                                                              #
        ################################################################################################################

        # grid represents the flattened LxL grid
        grid                          = np.zeros((self.L*self.L),dtype= TYPE1)
        # Add dysfunctional cells outside the structures
        dysfunctional_cmp             = np.zeros((self.L*self.L),dtype=np.bool)
        dysf_mask                     = (random_generator.uniform(0,1,self.L*self.L) <= delta_vector)
        dysfunctional_cmp[dysf_mask]  = True
        # Find the cmp structures
        cs_length_cmp, cs_paths_cmp   = mtd.check_structures_cmp_new(dysfunctional_cmp, candidate_paths, self.tau)
        # Calculate the number of dysfunctional cells
        n_dysf                        = np.count_nonzero(dysfunctional_cmp)
        # Construct a (1 x n_dysf) array containing the indexes of the dysfunctional cells
        dysf_loc                      = np.array(np.where(dysfunctional_cmp == 1)[0],dtype= TYPE1)
        # Store indicators for healthy cells
        healthy_cells                 = np.ones((self.L*self.L), dtype= np.bool)
        # left_links and right_links are boolean arrays telling whether a cell has a right/left connection
        left_links                    = np.ones((self.L*self.L), dtype= np.bool)
        right_links                   = np.ones((self.L*self.L), dtype= np.bool)
        # left_edge and right_edges are integer arrays containing the indexes of the cells lying in the edges
        left_edge                     = np.arange(0,self.L*self.L,self.L,dtype = TYPE1)
        right_edge                    = np.arange(self.L-1,self.L*self.L,self.L,dtype = TYPE1)
        top_edge                      = np.arange(0,self.L,1,dtype = TYPE1)
        bottom_edge                   = np.arange(self.L*self.L - self.L,self.L*self.L,1,dtype = TYPE1)
        # set the left links indicators on the left edge and the right links indicators on the right edge to FALSE
        left_links[left_edge]         = False
        right_links[right_edge]       = False
        # Track the number of active cells per step
        active_log                    = np.array(np.zeros(self.S),dtype=TYPE1)
        # Set the pacemaker type
        pacemaker_cells               = left_edge
        pacemaker_bool                = np.zeros((self.L*self.L), dtype= np.bool)
        # Create the dysfunctional matrix
        #dysf_matrix_cmp               = random_generator.uniform(0.0, 1.0, (self.S, n_dysf)) > self.epsilon
        # Set the active cells array
        active                        = (grid == self.tau + 1)
        # Initiate the timer
        begin                         = time.clock()
        # Loop over the S steps, starting from step number 2 as the first one simply consists in the left edge pacemaker
        for n in range(0,self.S,1):
                        grid -= 1

                        right_neighbors = mtd.shift2((active*right_links), 1)
                        left_neighbors  = mtd.shift2((active*left_links), -1)
                        down_neighbors  = mtd.shift2((active*down_links), self.L)
                        up_neighbors    = mtd.shift2((active*up_links), -self.L)

                        resting_cells   = grid < 1
                        potential_sites = (right_neighbors) | (left_neighbors)| (down_neighbors) | (up_neighbors)

                        temp_healthy              = np.copy(healthy_cells)
                        temp_healthy[dysf_loc]    = (random_generator.uniform(0.0, 1.0, n_dysf) > self.epsilon)
                        #temp_healthy[dysf_loc]   = dysf_matrix_cmp[n,:]


                        #grid -= 1


                        temp_pacemaker_bool = np.copy(pacemaker_bool)
                        temp_pacemaker_bool[pacemaker_cells] = (n % self.T == 0)*True

                        active                = ((resting_cells)*(potential_sites)*(temp_healthy)) | ((temp_pacemaker_bool)*(resting_cells))
                        grid[active]          = self.tau + 1

                        active_log[n]         = np.count_nonzero(active)

                        if active_log[n] != target_ts[n]:
                                print('Something went wrong in the replication')
                                break
                                return
                        else:
                                print('Step succesfully replicated')


        ############################################################
        #                Calculate AF, approach 1                  #
        ############################################################
        AF_boolean         = active_log >= self.t_af
        durations          = np.array(mtd.cmp_durations(AF_boolean))
        af_flag_cmp        = (np.count_nonzero(durations >= self.t_af) > 0)
        af_risk_cmp        = np.sum(AF_boolean)/float(self.S)
        active_log_cmp     = active_log

        # Terminate the timer
        end      = time.clock()
        # Prepare the output
        print('Standard CMP: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(self.nu, 2)) + ', Time in AF: ' + str(af_risk_cmp))
        print('Successfull replication')
        return [down_links, af_risk_cmp, af_flag_cmp, active_log_cmp, dysfunctional_cmp, cs_length_cmp, cs_paths_cmp]

    cpdef cmp_simulator_extension(self, target_ts, seed, simulation_id):

        # Set the seed
        random_generator = np.random.RandomState(seed)
        ################################################################################################################
        #                                                                                                              #
        #                                                Phase I                                                       #
        #                                                                                                              #
        ################################################################################################################
        cdef int n
        # the numpy array inputs are turned into 'C' array objects and conveniently renamed
        # grid represents the flattened LxL grid
        cdef np.ndarray grid          = np.zeros((self.L*self.L),dtype= TYPE1)
        # Track the primary critical structure
        dysfunctional,  down_links, cs_length_ccmp, cs_paths_ccmp, candidate_paths, delta_vector = mtd.build_grid_ccmp_new_rg(random_generator, self.nu,self.L,self.tau, self.delta)
        # Find the up links
        up_links                      = mtd.shift2(down_links, self.L)
        # Calculate the number of dysfunctional cells
        cdef int n_dysf               = np.count_nonzero(dysfunctional)
        # Construct a (1 x n_dysf) array containing the indexes of the dysfunctional cells
        cdef dysf_loc                 = np.array(np.where(dysfunctional == 1)[0],dtype= TYPE1)
        # Store indicators for healthy cells
        cdef np.ndarray healthy_cells = np.ones((self.L*self.L), dtype= np.bool)
        # left_links and right_links are boolean arrays telling whether a cell has a right/left connection
        cdef np.ndarray left_links    = np.ones((self.L*self.L), dtype= np.bool)
        cdef np.ndarray right_links   = np.ones((self.L*self.L), dtype= np.bool)
        # left_edge and right_edges are integer arrays containing the indexes of the cells lying in the edges
        cdef np.ndarray left_edge     = np.arange(0,self.L*self.L,self.L,dtype = TYPE1)
        cdef np.ndarray right_edge    = np.arange(self.L-1,self.L*self.L,self.L,dtype = TYPE1)
        cdef np.ndarray top_edge      = np.arange(0,self.L,1,dtype = TYPE1)
        cdef np.ndarray bottom_edge   = np.arange(self.L*self.L - self.L,self.L*self.L,1,dtype = TYPE1)
        # set the left links indicators on the left edge and the right links indicators on the right edge to FALSE
        left_links[left_edge]         = False
        right_links[right_edge]       = False
        # Track the number of active cells per step
        cdef np.ndarray active_log    = np.array(np.zeros(self.S),dtype=TYPE1)
        # Set the pacemaker type
        pacemaker_cells               = left_edge
        pacemaker_bool                = np.zeros((self.L*self.L), dtype= np.bool)
        # Create the dysfunctional matrix
        #dysf_matrix_ccmp              = random_generator.uniform(0.0, 1.0, (self.S, n_dysf)) > self.epsilon
        # Set the active cells array
        active                        = (grid == self.tau + 1)
        # Initiate the timer
        begin                         = time.clock()
        # Loop over the S steps, starting from step number 2 as the first one simply consists in the left edge pacemaker
        '''
        for n in range(0,self.S,1):
                            grid -= 1

                            right_neighbors = mtd.shift2((active*right_links), 1)
                            left_neighbors  = mtd.shift2((active*left_links), -1)
                            down_neighbors  = mtd.shift2((active*down_links), self.L)
                            up_neighbors    = mtd.shift2((active*up_links), -self.L)

                            resting_cells   = grid < 1
                            potential_sites = (right_neighbors) | (left_neighbors)| (down_neighbors) | (up_neighbors)

                            temp_healthy            = np.copy(healthy_cells)
                            temp_healthy[dysf_loc]  = (random_generator.uniform(0.0, 1.0, n_dysf) > self.epsilon)
                            #temp_healthy[dysf_loc]  = dysf_matrix_ccmp[n,:]


                            #grid -= 1

                            temp_pacemaker_bool = np.copy(pacemaker_bool)
                            temp_pacemaker_bool[pacemaker_cells] = (n % self.T == 0)*True

                            active                = ((resting_cells)*(potential_sites)*(temp_healthy)) | ((temp_pacemaker_bool)*(resting_cells))
                            grid[active]          = self.tau + 1

                            active_log[n]         = np.count_nonzero(active)



        ############################################################
        #                Calculate AF, approach 1                  #
        ############################################################
        AF_boolean         = active_log >= self.t_af
        durations          = np.array(mtd.cmp_durations(AF_boolean))
        af_flag_ccmp       = (np.count_nonzero(durations >= self.t_af) > 0)
        af_risk_ccmp       = np.sum(AF_boolean)/float(self.S)
        active_log_ccmp    = active_log
        dysfunctional_ccmp = dysfunctional
        # Terminate the timer
        '''
        end      = time.clock()
        # Prepare the output

        #print('Controlled CMP: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(self.nu, 2)) + ', Time in AF: ' + str(af_risk_ccmp))


        ################################################################################################################
        #                                                                                                              #
        #                                                Phase II                                                      #
        #                                                                                                              #
        ################################################################################################################

        # grid represents the flattened LxL grid
        grid                          = np.zeros((self.L*self.L),dtype= TYPE1)
        # Add dysfunctional cells outside the structures
        dysfunctional_cmp             = np.zeros((self.L*self.L),dtype=np.bool)
        dysf_mask                     = (random_generator.uniform(0,1,self.L*self.L) <= delta_vector)
        dysfunctional_cmp[dysf_mask]  = True
        # Find the cmp structures
        cs_length_cmp, cs_paths_cmp   = mtd.check_structures_cmp_new(dysfunctional_cmp, candidate_paths, self.tau)
        # Calculate the number of dysfunctional cells
        n_dysf                        = np.count_nonzero(dysfunctional_cmp)
        # Construct a (1 x n_dysf) array containing the indexes of the dysfunctional cells
        dysf_loc                      = np.array(np.where(dysfunctional_cmp == 1)[0],dtype= TYPE1)
        # Store indicators for healthy cells
        healthy_cells                 = np.ones((self.L*self.L), dtype= np.bool)
        # left_links and right_links are boolean arrays telling whether a cell has a right/left connection
        left_links                    = np.ones((self.L*self.L), dtype= np.bool)
        right_links                   = np.ones((self.L*self.L), dtype= np.bool)
        # left_edge and right_edges are integer arrays containing the indexes of the cells lying in the edges
        left_edge                     = np.arange(0,self.L*self.L,self.L,dtype = TYPE1)
        right_edge                    = np.arange(self.L-1,self.L*self.L,self.L,dtype = TYPE1)
        top_edge                      = np.arange(0,self.L,1,dtype = TYPE1)
        bottom_edge                   = np.arange(self.L*self.L - self.L,self.L*self.L,1,dtype = TYPE1)
        # set the left links indicators on the left edge and the right links indicators on the right edge to FALSE
        left_links[left_edge]         = False
        right_links[right_edge]       = False
        # Track the number of active cells per step
        active_log                    = np.array(np.zeros(self.S),dtype=TYPE1)
        # Set the pacemaker type
        pacemaker_cells               = left_edge
        pacemaker_bool                = np.zeros((self.L*self.L), dtype= np.bool)
        # Create the dysfunctional matrix
        #dysf_matrix_cmp               = random_generator.uniform(0.0, 1.0, (self.S, n_dysf)) > self.epsilon
        # Set the active cells array
        active                        = (grid == self.tau + 1)
        # Initiate the timer
        begin                         = time.clock()
        # Loop over the S steps, starting from step number 2 as the first one simply consists in the left edge pacemaker
        for n in range(0,self.S,1):
                        grid -= 1

                        right_neighbors = mtd.shift2((active*right_links), 1)
                        left_neighbors  = mtd.shift2((active*left_links), -1)
                        down_neighbors  = mtd.shift2((active*down_links), self.L)
                        up_neighbors    = mtd.shift2((active*up_links), -self.L)

                        resting_cells   = grid < 1
                        potential_sites = (right_neighbors) | (left_neighbors)| (down_neighbors) | (up_neighbors)

                        temp_healthy              = np.copy(healthy_cells)
                        temp_healthy[dysf_loc]    = (random_generator.uniform(0.0, 1.0, n_dysf) > self.epsilon)
                        #temp_healthy[dysf_loc]   = dysf_matrix_cmp[n,:]


                        #grid -= 1


                        temp_pacemaker_bool = np.copy(pacemaker_bool)
                        temp_pacemaker_bool[pacemaker_cells] = (n % self.T == 0)*True

                        active                = ((resting_cells)*(potential_sites)*(temp_healthy)) | ((temp_pacemaker_bool)*(resting_cells))
                        grid[active]          = self.tau + 1

                        active_log[n]         = np.count_nonzero(active)

                        if n < 10000000:
                                if active_log[n] != target_ts[n]:
                                        print('Something went wrong in the replication')
                                        break
                                        return
                                else:
                                        print('Step succesfully replicated')


        ############################################################
        #                Calculate AF, approach 1                  #
        ############################################################
        AF_boolean         = active_log >= self.t_af
        durations          = np.array(mtd.cmp_durations(AF_boolean))
        af_flag_cmp        = (np.count_nonzero(durations >= self.t_af) > 0)
        af_risk_cmp        = np.sum(AF_boolean)/float(self.S)
        active_log_cmp     = active_log

        # Terminate the timer
        end      = time.clock()
        # Prepare the output
        print('Standard CMP: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(self.nu, 2)) + ', Time in AF: ' + str(af_risk_cmp))
        print('Successfull replication')
        return [down_links, af_risk_cmp, af_flag_cmp, active_log_cmp, dysfunctional_cmp, cs_length_cmp, cs_paths_cmp]