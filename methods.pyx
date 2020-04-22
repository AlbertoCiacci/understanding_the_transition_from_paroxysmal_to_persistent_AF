import numpy as np
import random as rdn
from time import clock
from collections import Counter
import itertools
import matplotlib.pyplot as plt
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

##########################################################################
#                                                                        #
#                           Support Methods                              #
#                                                                        #
##########################################################################



cpdef shift2(array,lag):
    result = np.empty_like(array)
    n      = len(array)
    if lag > 0:
        result[:lag] = array[-lag:]
        result[lag:] = array[:(n-lag)]
    elif lag < 0:
        result[lag:] = array[0:np.abs(lag)]
        result[:lag] = array[np.abs(lag):]
    else:
        result = array
    return result

def moving_average(X, k):
    return [np.average(X[(i - k):i]) for i in range(k ,len(X)+1,1)]

def moving_std(X, k):
    return [np.std(X[(i - k):i]) for i in range(k ,len(X)+1,1)]

def check_functional(S, active_cells_ts, moving_average_ts, T, second_threshold, segment_size, repeats):

        idx_array = np.random.choice(np.arange(T - 1, S - segment_size, 1), repeats, replace = False)
        tag       = 0
        for rep in range(repeats):
                starting_idx = int(idx_array[rep])
                final_idx    = int(starting_idx + segment_size)
                condition_1  = np.count_nonzero(active_cells_ts[starting_idx:final_idx] < T)
                condition_2  = (np.max(moving_average_ts[starting_idx:final_idx]) - np.min(moving_average_ts[starting_idx:final_idx]))  >= second_threshold
                if (condition_1 == 0) & (condition_2 == False):
                    tag = 1
                    break
        return tag


def initiate_grid(L, tau, T, vlinks_x, vlinks_y, dysf_loc, scheme_min_x, scheme_max_x, scheme_min_y, scheme_max_y, grid, ax):
        # Plot pacemaker
        #pacemaker_line    = ax.vlines(0%T,0,L,color='red')
        # Plot longitudinal connections
        for idx in range(L):
                ax.plot(np.arange(0,L,1), np.repeat(idx,L), color = 'k')
        # Plot vertical connections
        toPlot = zip(vlinks_x,vlinks_y, vlinks_x, vlinks_y + 1)
        for tuple in toPlot:
                ax.plot([tuple[0], tuple[2]], [(L - 1) - tuple[1], (L - 1) - tuple[3]],color='k')
        # Plot dysfunctional cells
        #ax.scatter(np.array(dysf_loc)%L,(L - 1) - np.floor(np.array(dysf_loc)/float(L)),linewidth = 3,facecolors = 'none',edgecolors = 'red',s = 250, marker='s',zorder = 12)
        ax.scatter(np.array(dysf_loc)%L,(L - 1) - np.floor(np.array(dysf_loc)/float(L)),linewidth = 3,facecolors = 'none',edgecolors = 'red',s = 125, marker='s',zorder = 12)
        # Set the ticks and labels in the axis
        ax.set_xticks(np.arange(0,200,10))
        ax.set_yticks(np.arange(0,200,5))
        ax.set_yticklabels(np.arange(200,0,-5))
        ax.set_xticklabels(np.arange(0,200,10))
        ax.tick_params(axis=u'both', which=u'both',length=0)
        # Set the plot limits
        ax.set_xlim([scheme_min_x -1,scheme_max_x + 1])
        ax.set_ylim([(L - 1) - scheme_max_y - 0.5,(L - 1) - scheme_min_y + 0.5])
        # Set the axis labels
        ax.set_xlabel('Longitudinal (' + r'$\nu_{\parallel}$' + ')',fontsize = 16)
        ax.set_ylabel('Transversal (' + r'$\nu_{\perp}$' + ')',fontsize = 16)
        # Plot the cells
        x    = np.array(list(np.arange(0,L,1))*L)
        y    = L - 1 - np.repeat(np.arange(0,L,1),L)
        c    = [list(np.repeat(np.maximum(el,0)/float(tau + 1),3)) for el in grid]
        #scat = plt.scatter(x,y,linewidth = 1,facecolors = c,edgecolors = 'k',s = 150, marker='s',zorder = 10)
        scat = plt.scatter(x,y,linewidth = 1,facecolors = c,edgecolors = 'k',s = 75, marker='s',zorder = 10)
        pml, = plt.plot([0%T,0%T],[0,L],color='red', linewidth = 1, zorder = 20)
        #ax.yaxis.set_major_locator(plt.NullLocator())
        return scat, ax, pml

def plot_grid(ax, figure_tag, L, tau, T, vlinks_x, vlinks_y, dysf_loc, scheme_min_x, scheme_max_x, scheme_min_y, scheme_max_y, colors, t):
        #ax.axis('equal')
        for idx in range(L):
                ax.plot(np.arange(0,L,1), np.repeat(idx,L), color = 'k')
        # Plot vertical connections
        toPlot = zip(vlinks_x,vlinks_y, vlinks_x, vlinks_y + 1)
        for tuple in toPlot:
                ax.plot([tuple[0], tuple[2]], [(L - 1) - tuple[1], (L - 1) - tuple[3]],color='k')
        # Plot dysfunctional cells
        ax.scatter(np.array(dysf_loc)%L,(L - 1) - np.floor(np.array(dysf_loc)/float(L)),linewidth = 1,facecolors = 'none',edgecolors = 'red',s = 160, marker='s',zorder = 12)
        # Set the ticks and labels in the axis
        ax.set_xticks(np.arange(0,200,10))
        ax.set_yticks(np.arange(0,200,5))
        ax.set_yticklabels(np.arange(200,0,-5),fontsize = 17)
        ax.set_xticklabels(np.arange(0,200,10),fontsize = 17)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        # Set the plot limits
        ax.set_xlim([scheme_min_x - 0.5,scheme_max_x + 0.5])
        ax.set_ylim([(L - 1) - scheme_max_y - 0.5,(L - 1) - scheme_min_y + 0.5])
        # Set the axis labels
        #ax.set_xlabel('Longitudinal (' + r'$\nu_{\parallel}$' + ')',fontsize = 18)
        #ax.set_ylabel('Transversal (' + r'$\nu_{\perp}$' + ')',fontsize = 18)
        #ax.set_title("t: " + str(t), fontsize = 18)
        # Plot the cells
        x    = np.array(list(np.arange(0,L,1))*L)
        y    = L - 1 - np.repeat(np.arange(0,L,1),L)
        ax.scatter(x,y,linewidth = 1,facecolors = colors,edgecolors = 'k',s = 140, marker='s',zorder = 10)
        ax.plot([t%T,t%T],[0,L],color='red', linewidth = 1, zorder = 20)
        #ax.yaxis.set_major_locator(plt.NullLocator())
        ax.text(-3,(L - 1) - scheme_min_y + 0.35, figure_tag, horizontalalignment='center',verticalalignment = 'center',fontsize=22)



        '''
        for idx in range(L):
                ax.plot(np.arange(0,L,1), np.repeat(idx,L), color = 'k')
        # Plot vertical connections
        toPlot = zip(vlinks_x,vlinks_y, vlinks_x, vlinks_y + 1)
        for tuple in toPlot:
                ax.plot([tuple[0], tuple[2]], [(L - 1) - tuple[1], (L - 1) - tuple[3]],color='k')
        # Plot dysfunctional cells
        ax.scatter(np.array(dysf_loc)%L,(L - 1) - np.floor(np.array(dysf_loc)/float(L)),linewidth = 1,facecolors = 'none',edgecolors = 'red',s = 160, marker='s',zorder = 12)
        # Set the ticks and labels in the axis
        ax.set_xticks(np.arange(0,200,10))
        ax.set_yticks(np.arange(0,200,5))
        ax.set_yticklabels(np.arange(200,0,-5),fontsize = 22)
        ax.set_xticklabels(np.arange(0,200,10),fontsize = 22)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        # Set the plot limits
        ax.set_xlim([scheme_min_x - 0.5,scheme_max_x + 0.5])
        ax.set_ylim([(L - 1) - scheme_max_y - 0.25,(L - 1) - scheme_min_y + 0.25])
        # Set the axis labels
        #ax.set_xlabel('Longitudinal (' + r'$\nu_{\parallel}$' + ')',fontsize = 18)
        #ax.set_ylabel('Transversal (' + r'$\nu_{\perp}$' + ')',fontsize = 18)
        #ax.set_title("t: " + str(t), fontsize = 18)
        # Plot the cells
        x    = np.array(list(np.arange(0,L,1))*L)
        y    = L - 1 - np.repeat(np.arange(0,L,1),L)
        ax.scatter(x,y,linewidth = 1,facecolors = colors,edgecolors = 'k',s = 140, marker='s',zorder = 10)
        ax.plot([t%T,t%T],[0,L],color='red', linewidth = 1, zorder = 20)
        ax.text(-3,(L - 1) - scheme_min_y + 0.35, figure_tag, horizontalalignment='center',verticalalignment = 'center',fontsize=22)
        '''


def cmp_durations(AF_timesteps):
    durations = [ sum( 1 for _ in group ) for key, group in itertools.groupby( AF_timesteps ) if key ]
    return durations



cpdef AF_tracker(S, activity_tracker, cs_paths, cs_dysf, cs_sites, cs_length):
        # Count the number of structures
        n_cs                       = len(cs_paths)
        # Find the slicing indexes
        low_temp                   = 0
        high_temp                  = 0
        low_indices                = []
        high_indices               = []
        for i in range(0,n_cs,1):
            high_temp = low_temp + cs_length[i]
            low_indices.append(low_temp)
            high_indices.append(high_temp)
            low_temp  = high_temp

        # Store the activity of each structure
        structure_activity_monitor = np.zeros((n_cs, S), dtype = np.int)
        # Check whether each activation of dysfunctional cells lead to a chain
        for i in range(0,n_cs,1):
                copy_tracker           = np.copy(activity_tracker)
                sliced_tracker         = copy_tracker[low_indices[i]:high_indices[i],:]
                # Find all the activation times
                dysf_loc               = cs_paths[i].index(cs_dysf[i])
                dysf_activation_times  = np.where(sliced_tracker[dysf_loc,:] == True)[0]
                # Compute the index progressions
                path_a_idx             = [dysf_loc] + list(range(dysf_loc+1,len(cs_paths[i]),1)) + list(range(0,dysf_loc,1))
                path_b_idx             = [dysf_loc] + list(range(dysf_loc-1,-1,-1)) + list(range(len(cs_paths[i]) - 1,dysf_loc,-1))
                # Analyze each activation time
                for t in dysf_activation_times:
                    # Check direction A
                    progression_counter_a = 0
                    for idx in path_a_idx:
                        target = sliced_tracker[idx,t+progression_counter_a]
                        if target == True:
                                  progression_counter_a += 1
                        else:
                                  break
                        if t + progression_counter_a >= S:
                                  break

                    # Check direction B
                    progression_counter_b = 0
                    for idx in path_b_idx:
                        target = sliced_tracker[idx,t+progression_counter_b]
                        if target == True:
                                  progression_counter_b += 1
                        else:
                                  break
                        if t + progression_counter_b >= S:
                                  break


                    if (progression_counter_a == len(path_a_idx)):
                          structure_activity_monitor[i,t:(t+progression_counter_a)] = 1

                    if (progression_counter_b == len(path_b_idx)):
                          structure_activity_monitor[i,t:(t+progression_counter_b)] = 1

        active_structures_timeline = np.sum(structure_activity_monitor,axis = 0)
        AF_timestamps              = active_structures_timeline > 0
        af_risk                    = np.sum(AF_timestamps)/float(S)
        durations                  = cmp_durations(AF_timestamps)

        # Prepare the structure matrix
        sorted_indexes             = list(np.argsort(cs_length))
        monitor_matrix             = structure_activity_monitor[[sorted_indexes]]


        return active_structures_timeline, AF_timestamps, af_risk, durations, monitor_matrix


cpdef build_grid_ccmp_new(target_nu,L,tau, delta):

      # Store segments
      candidate_segments = []
      effective_segments = []
      structures_length  = []
      # Random boolean matrix for down links
      down_links_matrix    = (np.random.uniform(0,1,(L,L)) <= target_nu)
      # Random boolean matrix for down links
      dysfunctional_matrix = np.zeros((L, L), dtype = np.bool)
      # Set up the delta vector
      delta_vector         = np.repeat(delta,L*L)
      # Rows and columns of transversal downward links
      rows, cols           = np.where(down_links_matrix == True)
      for row in np.unique(rows):
                  # Find the boolean of the down and up links
                  down_links_boolean    = (down_links_matrix[row,:] == True)
                  up_links_boolean      = (down_links_matrix[((row - 1)%L),:] == True)
                  # Find the locations of transversal links
                  trans_links_locations = np.where(down_links_boolean | up_links_boolean)[0]
                  # Find the distance between adjacent vertical links
                  segment_lengths       = np.diff(trans_links_locations)
                  # Segments >= tau/2 becomes eligible, so we find their indexes in the location vector above
                  potential_candidates  = np.where(segment_lengths >= np.int(0.5*tau))[0]
                  # Find the starting points
                  starting_points       = trans_links_locations[potential_candidates]
                  # Find the final points
                  final_points          = starting_points + segment_lengths[potential_candidates]
                  # Count the number of candidate points
                  n_candidates          = len(starting_points)
                  # Loop over the candidates
                  for n in range(n_candidates):
                            # Find the x-coordinates of left edge of the segment
                            starting_x      = starting_points[n]
                            # Find the x-coordinates of right edge of the segment
                            final_x         = final_points[n]
                            # Looking from the right edge, find the first available site that should host a dysfunctional site to create a simple structure
                            first_eligible  = final_x - np.int(0.5*tau) + 1  ##Changed
                            # Try to place a dysfunctional cell in the segment
                            if first_eligible > starting_x:
                                        n_dysf_placed = 0
                                        for position in range(first_eligible, starting_x, -1):
                                                if (np.random.uniform(0, 1) <= delta):
                                                        n_dysf_placed = 1
                                                        dysfunctional_matrix[row, position] = True
                                                        break

                                        segment = list(L*row + np.arange(starting_x, final_x + 1, 1))
                                        # If a dysfunctional has been placed, store the segment
                                        if n_dysf_placed == 1:
                                                effective_segments.append(segment)
                                                structures_length.append(2*len(segment))

                                        candidate_segments.append(segment)
                                        delta_vector[segment] = -1.0


      down_links    = np.reshape(down_links_matrix, L*L)
      dysfunctional = np.reshape(dysfunctional_matrix, L*L)
      delta_vector[dysfunctional] = 2.0
      return dysfunctional,  down_links, structures_length, effective_segments, candidate_segments, delta_vector

cpdef build_grid_ccmp_new_rg(random_generator, target_nu,L,tau, delta):

      # Store segments
      candidate_segments = []
      effective_segments = []
      structures_length  = []
      # Random boolean matrix for down links
      down_links_matrix    = (random_generator.uniform(0,1,(L,L)) <= target_nu)
      # Random boolean matrix for down links
      dysfunctional_matrix = np.zeros((L, L), dtype = np.bool)
      # Set up the delta vector
      delta_vector         = np.repeat(delta,L*L)
      # Rows and columns of transversal downward links
      rows, cols           = np.where(down_links_matrix == True)
      for row in np.unique(rows):
                  # Find the boolean of the down and up links
                  down_links_boolean    = (down_links_matrix[row,:] == True)
                  up_links_boolean      = (down_links_matrix[((row - 1)%L),:] == True)
                  # Find the locations of transversal links
                  trans_links_locations = np.where(down_links_boolean | up_links_boolean)[0]
                  # Find the distance between adjacent vertical links
                  segment_lengths       = np.diff(trans_links_locations)
                  # Segments >= tau/2 becomes eligible, so we find their indexes in the location vector above
                  potential_candidates  = np.where(segment_lengths >= np.int(0.5*tau))[0]
                  # Find the starting points
                  starting_points       = trans_links_locations[potential_candidates]
                  # Find the final points
                  final_points          = starting_points + segment_lengths[potential_candidates]
                  # Count the number of candidate points
                  n_candidates          = len(starting_points)
                  # Loop over the candidates
                  for n in range(n_candidates):
                            # Find the x-coordinates of left edge of the segment
                            starting_x      = starting_points[n]
                            # Find the x-coordinates of right edge of the segment
                            final_x         = final_points[n]
                            # Looking from the right edge, find the first available site that should host a dysfunctional site to create a simple structure
                            first_eligible  = final_x - np.int(0.5*tau) + 1  ##Changed
                            # Try to place a dysfunctional cell in the segment
                            if first_eligible > starting_x:
                                        n_dysf_placed = 0
                                        for position in range(first_eligible, starting_x, -1):
                                                if (random_generator.uniform(0, 1) <= delta):
                                                        n_dysf_placed = 1
                                                        dysfunctional_matrix[row, position] = True
                                                        break

                                        segment = list(L*row + np.arange(starting_x, final_x + 1, 1))
                                        # If a dysfunctional has been placed, store the segment
                                        if n_dysf_placed == 1:
                                                effective_segments.append(segment)
                                                structures_length.append(2*len(segment))

                                        candidate_segments.append(segment)
                                        delta_vector[segment] = -1.0


      down_links    = np.reshape(down_links_matrix, L*L)
      dysfunctional = np.reshape(dysfunctional_matrix, L*L)
      delta_vector[dysfunctional] = 2.0
      return dysfunctional,  down_links, structures_length, effective_segments, candidate_segments, delta_vector

cpdef check_structures_cmp_new(dysfunctional, candidate_segments, tau):
      effective_segments = []
      structures_length  = []
      for segment in candidate_segments:
          starting_x      = segment[0]
          final_x         = segment[-1]
          first_eligible  = final_x - np.int(0.5*tau) + 1  ##Changed
          n_dysf_right    = np.sum(dysfunctional[(first_eligible + 1):(final_x + 1)])
          n_dysf_left     = np.sum(dysfunctional[(starting_x + 1):(first_eligible + 1)])
          n_dysf_start    = dysfunctional[starting_x]
          if (n_dysf_right == 0) & (n_dysf_start == 0) & (n_dysf_left == 1):
                         effective_segments.append(segment)
                         structures_length.append(2*len(segment))

      return structures_length, effective_segments

'''
cpdef cmp_structure_matching_test(down_links, L, dysfunctional, candidate_segments, tau):
      # Store segments
      candidate_segments = []
      effective_segments = []
      structures_length  = []
      # Random boolean matrix for down links
      down_links_matrix    = np.reshape(down_links,(L,L))
      # Rows and columns of transversal downward links
      rows, cols           = np.where(down_links_matrix == True)
      for row in np.unique(rows):
                  # Find the boolean of the down and up links
                  down_links_boolean    = (down_links_matrix[row,:] == True)
                  up_links_boolean      = (down_links_matrix[((row - 1)%L),:] == True)
                  # Find the locations of transversal links
                  trans_links_locations = np.where(down_links_boolean | up_links_boolean)[0]
                  # Find the distance between adjacent vertical links
                  segment_lengths       = np.diff(trans_links_locations)
                  # Segments >= tau/2 becomes eligible, so we find their indexes in the location vector above
                  potential_candidates  = np.where(segment_lengths >= np.int(0.5*tau))[0]
                  # Find the starting points
                  starting_points       = trans_links_locations[potential_candidates]
                  # Find the final points
                  final_points          = starting_points + segment_lengths[potential_candidates]
                  # Count the number of candidate points
                  n_candidates          = len(starting_points)
                  # Loop over the candidates
                  for n in range(n_candidates):
                            # Find the x-coordinates of left edge of the segment
                            starting_x      = starting_points[n]
                            # Find the x-coordinates of right edge of the segment
                            final_x         = final_points[n]
                            # Looking from the right edge, find the first available site that should host a dysfunctional site to create a simple structure
                            first_eligible  = final_x - np.int(0.5*tau) + 1  ##Changed
                            # Try to place a dysfunctional cell in the segment
                            if first_eligible > starting_x:
                                        n_dysf_placed = 0
                                        for position in range(first_eligible, starting_x, -1):
                                                current_id = L*row + position
                                                if dysfunctional[current_id] == True:
                                                    n_dysf_placed += 1
                                        if n_dysf_placed > 1:
                                                print('Problem found, we cannot have two or more dysfunctional cells here')

                                        segment = list(L*row + np.arange(starting_x, final_x + 1, 1))
                                        # If a dysfunctional has been placed, store the segment
                                        if n_dysf_placed == 1:
                                                effective_segments.append(segment)
                                                structures_length.append(2*len(segment))

                                        candidate_segments.append(segment)
                                        delta_vector[segment] = -1.0


      down_links    = np.reshape(down_links_matrix, L*L)
      dysfunctional = np.reshape(dysfunctional_matrix, L*L)
      delta_vector[dysfunctional] = 2.0
      return dysfunctional,  down_links, structures_length, effective_segments, candidate_segments, delta_vector
'''
