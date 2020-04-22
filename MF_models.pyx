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


cdef class MF_models:
    cdef int L, S, T, scaling_factor
    cdef double epsilon

    def __init__(self, L, S, T, epsilon, scaling_factor):
            # This is equivalent to a C++ constructor

            self.L                    = L                       # Grid size
            self.S                    = S                       # Number of steps
            self.T                    = T                       # Pacemaker period
            self.epsilon              = epsilon                 # Failure rate for dysfunctional cells
            self.scaling_factor       = scaling_factor          # Set the scaling of the system by multiplying the number of structures


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
    cpdef mf_simulator(self, seed, cs_length, nu, simulation_id):
        random_generator = np.random.RandomState(seed)

        #########################################################
        #                Standard MF Model                      #
        #########################################################

        begin = time.clock()
        n_cs  = len(cs_length)
        if n_cs < 1:
                    # Terminate the timer
                    end                    = time.clock()
                    af_flag_MF             = False
                    af_risk_MF             = 0.0
                    n_active_particles_MF  = np.array(np.zeros(self.S),dtype=TYPE1)
                    print('Standard MF: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(nu, 2)))
        else:
                    # Measure the number of active particles
                    n_active_particles_MF  = np.array(np.zeros(self.S),dtype=TYPE1)
                    # Set the probabilities
                    p                      =  self.epsilon/float(np.average(cs_length))
                    q                      =  self.epsilon/float(np.average(cs_length))
                    p0                     =  self.epsilon/float(self.T)
                    # Set the status array
                    states                 = - np.ones(n_cs)
                    n_active               = np.count_nonzero(states == 1)
                    # Perform S steps of a Markov Chain
                    for n in range(0, self.S, 1):
                        # Store the number of active particles
                        n_active_particles_MF[n] = n_active
                        # Allow targeted particles to potentially switch their statuses
                        if n_active == 0:
                             random_drawn         = (random_generator.uniform(0,1,n_cs) <= p0)
                        else:
                             random_drawn         = (random_generator.uniform(0,1,n_cs) <= p)
                        states[random_drawn] *= -1
                        # Count the post-update number of active particles
                        n_active             = np.count_nonzero(states == 1)



                    # Calculate the durations and risk
                    AF_timesteps = n_active_particles_MF > 0
                    af_flag_MF   = (np.count_nonzero(AF_timesteps) > 0)
                    af_risk_MF   = np.sum(AF_timesteps)/float(self.S)
                    # Terminate the timer
                    end = time.clock()
                    # Prepare the output
                    print('Standard MF: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(nu, 2)))


        #########################################################
        #                Enhanced MF Model                      #
        #########################################################

        begin = time.clock()

        if n_cs == 0:
                    # Terminate the timer
                    end                    = time.clock()
                    af_flag_eMF             = False
                    af_risk_eMF             = 0.0
                    n_active_particles_eMF  = np.array(np.zeros(self.S),dtype=TYPE1)
                    print('Enhanced MF: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(nu, 2)))
        else:

                    # Measure the number of active particles
                    n_active_particles_eMF  = np.array(np.zeros(self.S),dtype=TYPE1)
                    # Set the probabilities
                    lengths                 = (np.array(cs_length)).astype(float)
                    # Set the arrays to check the times
                    check_times             = np.random.randint(0,self.L, n_cs)
                    # Set the status array
                    states                  = -np.ones(n_cs)
                    n_active                = np.count_nonzero(states == 1)
                    # Perform S steps of a Markov Chain
                    for n in range(0, self.S, 1):
                                # Store the number of active particles
                                n_active_particles_eMF[n] = n_active
                                reacting_particles        = (check_times == n)
                                random_drawn              = (random_generator.uniform(0,1,n_cs) <= self.epsilon)
                                switching                 = (reacting_particles) & (random_drawn)
                                states[switching]        *= -1.0
                                n_active                  = np.count_nonzero(states == 1)
                                if n_active == 0:
                                        check_times[reacting_particles] += self.T
                                else:
                                        active_bool                                        = (states == 1)
                                        inactive_bool                                      = ~active_bool
                                        dominating_length                                  =  np.min(lengths[active_bool])
                                        check_times[(active_bool & reacting_particles)]    = (check_times[(active_bool & reacting_particles)] + lengths[(active_bool & reacting_particles)])
                                        check_times[(inactive_bool & reacting_particles)]  = (check_times[(inactive_bool & reacting_particles)] + dominating_length)


                    # Calculate the durations and risk
                    AF_timesteps     = n_active_particles_eMF > 0
                    af_flag_eMF      = (np.count_nonzero(AF_timesteps) > 0)
                    af_risk_eMF      = np.count_nonzero(AF_timesteps)/float(self.S)
                    # Terminate the timer
                    end = time.clock()
                    print('Enhanced MF: Processing time: ' + str(end - begin) +' secs., Simulation ID : ' + str(simulation_id + 1) + ', nu: ' + str(np.round(nu, 2)))

        #########################################################
        #                Continuous MF Model                    #
        #########################################################
        av_l             = np.average(cs_length)
        if n_cs > 0:
                    af_risk_cMF          =  (2**(n_cs) - 1.0)/float(2**(n_cs) - 1.0 + self.T/float(av_l))
        else:
                    af_risk_cMF          = 0.0
        print('Continuous MF: calculation complete. Simulation ID : ' + str(simulation_id + 1) + ', nu: ' + str(np.round(nu, 2)))

        #########################################################
        #                Scaled MF Model                        #
        #########################################################

        begin = time.clock()
        n_cs             = np.int(self.scaling_factor*len(cs_length))
        if n_cs == 0:
             # Terminate the timer
             end                     = time.clock()
             af_flag_2MF             = False
             af_risk_2MF             = 0.0
             n_active_particles_2MF  = np.array(np.zeros(self.S),dtype=TYPE1)
             print('Scaled MF: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', scaling factor: ' + str(self.scaling_factor) + ', nu: ' + str(np.round(nu, 2)))
        else:
            n_active_particles_2MF   =  np.array(np.zeros(self.S),dtype=TYPE1)
            avg_l                    =  np.average(np.repeat(cs_length, self.scaling_factor))
            p                        =  self.epsilon/float(avg_l)
            q                        =  self.epsilon/float(avg_l)
            p0                       =  self.epsilon/float(self.T)
            # Set the status array
            states                   = - np.ones(n_cs)
            n_active                 = np.count_nonzero(states == 1)
            # Perform S steps of a Markov Chain
            for n in range(0, self.S, 1):
                # Store the number of active particles
                n_active_particles_2MF[n] = n_active
                # Allow targeted particles to potentially switch their statuses
                if n_active == 0:
                     random_drawn         = (random_generator.uniform(0,1,n_cs) <= p0)
                else:
                     random_drawn         = (random_generator.uniform(0,1,n_cs) <= p)
                states[random_drawn] *= -1
                # Count the post-update number of active particles
                n_active             = np.count_nonzero(states == 1)



            # Calculate the durations and risk
            AF_timesteps = n_active_particles_2MF > 0
            af_flag_2MF  = (np.count_nonzero(AF_timesteps) > 0)
            af_risk_2MF  = np.sum(AF_timesteps)/float(self.S)
            # Terminate the timer
            end = time.clock()
            # Prepare the output
            print('Scaled MF: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', scaling factor: ' + str(self.scaling_factor) + ', nu: ' + str(np.round(nu, 2)))


        return [[af_risk_MF, af_flag_MF, n_active_particles_MF],[af_risk_eMF, af_flag_eMF, n_active_particles_eMF],[af_risk_cMF],[af_risk_2MF, af_flag_2MF, n_active_particles_2MF]]





