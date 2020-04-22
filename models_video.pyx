
######################################################
#               Graphic Parameters                   #
######################################################
pic_width         = 400                     # Set the width (px) of the dynamic plot
pic_height        = 400                     # Set the height (px) of the dynamic plot
wait_ms           = 1                       # Set the waiting time (ms) between two consecutive displayed pictures
fps               = 20                      # Set the fps of the video
alpha             = 0.4                     # Transparency factor
video_format      = '.avi'                  # Set the video format


######################################################################
#                                                                    #
#                     Import Libraries                               #
#                                                                    #
######################################################################
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import cv2
plt.rcParams['animation.ffmpeg_path'] = "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"
from matplotlib import animation
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
    cdef int L, S, T, tau, starting_capture_time, ending_capture_time, scheme_min_x, scheme_max_x, scheme_min_y, scheme_max_y, produce_videos_1, produce_schemes_1, produce_videos_2, produce_schemes_2, fps, interval
    cdef double delta, epsilon, nu

    def __init__(self, L, S, T, tau, delta, epsilon, nu, starting_capture_time, ending_capture_time, scheme_min_x, scheme_max_x, scheme_min_y, scheme_max_y, produce_videos_1, produce_schemes_1, produce_videos_2, produce_schemes_2, fps, interval):
            # This is equivalent to a C++ constructor

            self.L                     = L                       # Grid size
            self.S                     = S                       # Number of steps
            self.T                     = T                       # Pacemaker period
            self.tau                   = tau                     # Refractory period
            self.delta                 = delta                   # Dysfunctional cell ratio
            self.epsilon               = epsilon                 # Failure rate for dysfunctional cells
            self.nu                    = nu                      # Transversal connections rate
            self.starting_capture_time = starting_capture_time
            self.ending_capture_time   = ending_capture_time
            self.scheme_min_x          = scheme_min_x
            self.scheme_max_x          = scheme_max_x
            self.scheme_min_y          = scheme_min_y
            self.scheme_max_y          = scheme_max_y
            self.produce_videos_1      = produce_videos_1
            self.produce_schemes_1     = produce_schemes_1
            self.produce_videos_2      = produce_videos_2
            self.produce_schemes_2     = produce_schemes_2
            self.fps                   = fps
            self.interval              = interval


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
    def cmp_simulator(self, output_folder, target_frames, cCMP_folder, CMP_folder, simulation_id):
        # Set the video name
        video_name                    = 'video_' + str(simulation_id)
        schematic_name                = 'schematic_' + str(simulation_id)

        if (self.produce_schemes_1 == 1) | (self.produce_videos_1 == 1):

                        # Load files
                        dysfunctional                 = np.load(cCMP_folder + "experiment_" + str(simulation_id) + "\\dysfunctional_cCMP_" + str(simulation_id) + '.npy')
                        down_links                    = np.load(cCMP_folder + "experiment_" + str(simulation_id) + "\\down_links_" + str(simulation_id) + '.npy')
                        up_links                      = mtd.shift2(down_links, self.L)
                        nodes_per_step_target         = np.load(cCMP_folder + "experiment_" + str(simulation_id) + "\\active_nodes_cCMP_" + str(simulation_id) + '.npy')
                        cs_paths_cCMP                 = np.load(cCMP_folder + "experiment_" + str(simulation_id) + "\\cs_paths_cCMP_" + str(simulation_id) + '.npy')
                        dysf_matrix_ccmp              = (np.load(cCMP_folder + "experiment_" + str(simulation_id) + "\\dysf_matrix_cCMP_" + str(simulation_id) + '.npz'))['arr_0']
                        ################################################################################################################
                        #                                                                                                              #
                        #                                                Phase I                                                       #
                        #                                                                                                              #
                        ################################################################################################################

                        ################################################################################################################
                        #                                             Model Input                                                      #
                        ################################################################################################################

                        # the numpy array inputs are turned into 'C' array objects and conveniently renamed
                        # grid represents the flattened LxL grid
                        grid          = np.zeros((self.L*self.L),dtype= TYPE1)
                        # Calculate the number of dysfunctional cells
                        n_dysf        = np.count_nonzero(dysfunctional)
                        # Construct a (1 x n_dysf) array containing the indexes of the dysfunctional cells
                        dysf_loc      = np.array(np.where(dysfunctional == 1)[0],dtype= TYPE1)
                        # Store indicators for healthy cells
                        healthy_cells = np.ones((self.L*self.L), dtype= np.bool)
                        # left_links and right_links are boolean arrays telling whether a cell has a right/left connection
                        left_links    = np.ones((self.L*self.L), dtype= np.bool)
                        right_links   = np.ones((self.L*self.L), dtype= np.bool)
                        # left_edge and right_edges are integer arrays containing the indexes of the cells lying in the edges
                        left_edge     = np.arange(0,self.L*self.L,self.L,dtype = TYPE1)
                        right_edge    = np.arange(self.L-1,self.L*self.L,self.L,dtype = TYPE1)
                        top_edge      = np.arange(0,self.L,1,dtype = TYPE1)
                        bottom_edge   = np.arange(self.L*self.L - self.L,self.L*self.L,1,dtype = TYPE1)
                        # set the left links indicators on the left edge and the right links indicators on the right edge to FALSE
                        left_links[left_edge]         = False
                        right_links[right_edge]       = False
                        # Set the pacemaker type
                        pacemaker_cells   = left_edge
                        pacemaker_bool    = np.zeros((self.L*self.L), dtype= np.bool)
                        # Set the active cells array
                        active            = (grid == self.tau + 1)
                        ################################################################################################################
                        #                                             Video Input                                                      #
                        ################################################################################################################
                        if (self.produce_videos_1 == 1):
                                    video_directory  = output_folder + "\\animations\\videos\\"+ "{:.2f}".format(self.nu)+"\\cCMP\\experiment_" + str(simulation_id)
                                    os.makedirs(video_directory, exist_ok=True)
                                    cap    = cv2.VideoCapture(0)
                                    x_lims = [self.scheme_min_x, self.scheme_max_x]
                                    y_lims = [self.scheme_min_y, self.scheme_max_y]
                                    out    = cv2.VideoWriter(video_directory + "\\" + video_name + video_format, -1, fps, (pic_width,pic_height))
                        ################################################################################################################
                        #                                        Stylized Grid Initiation                                              #
                        ################################################################################################################
                        if (self.produce_schemes_1 == 1):
                                    # Set up the video writer and the target folder
                                    grid_container    = []
                                    scheme_directory  =  output_folder + "\\animations\\schematics\\"+ "{:.2f}".format(self.nu)+"\\cCMP\\experiment_" + str(simulation_id)
                                    os.makedirs(scheme_directory, exist_ok=True)
                                    # Linearize vertical links
                                    vlinks_x          = (np.where(down_links == True)[0])%self.L
                                    vlinks_y          =  np.floor((np.where(down_links == True)[0])/float(self.L))
                                    # Open the figure object
                                    fig, ax           = plt.subplots(figsize=(15,7.5),dpi=200)
                                    # Initiate the grid
                                    scat, ax, pml     = mtd.initiate_grid(self.L,self.tau,self.T,vlinks_x,vlinks_y,dysf_loc,self.scheme_min_x, self.scheme_max_x, self.scheme_min_y, self.scheme_max_y, grid, ax)

                                    def init():
                                        return ax

                                    def animate(frame_id,scat,color_list,T,L,t_span,ax,pml):
                                        scat.set_facecolors(color_list[frame_id])
                                        scat.set_edgecolors('k')
                                        pml.set_data([t_span[frame_id]%T,t_span[frame_id]%T],[0,L])
                                        ax.set_title("t: " + str(t_span[frame_id]), fontsize = 18)
                                        print('Animation for timestep: ' + str(t_span[frame_id]) + ' completed.')
                                        return scat, pml,

                        ################################################################################################################
                        #                                             CMP Algorithm                                                    #
                        ################################################################################################################
                        begin         = time.clock()
                        # Loop over the S steps, starting from step number 2 as the first one simply consists in the left edge pacemaker
                        for n in range(0,self.S,1):
                                        grid -= 1

                                        right_neighbors = mtd.shift2((active*right_links), 1)
                                        left_neighbors  = mtd.shift2((active*left_links), -1)
                                        down_neighbors  = mtd.shift2((active*down_links), self.L)
                                        up_neighbors    = mtd.shift2((active*up_links), -self.L)

                                        resting_cells   = grid < 1
                                        potential_sites = (right_neighbors) | (left_neighbors)| (down_neighbors) | (up_neighbors)

                                        temp_healthy            = np.copy(healthy_cells)
                                        temp_healthy[dysf_loc]  = dysf_matrix_ccmp[n,:]


                                        temp_pacemaker_bool = np.copy(pacemaker_bool)
                                        temp_pacemaker_bool[pacemaker_cells] = (n % self.T == 0)*True

                                        active                = ((resting_cells)*(potential_sites)*(temp_healthy)) | ((temp_pacemaker_bool)*(resting_cells))
                                        grid[active]          = self.tau + 1

                                        if np.count_nonzero(active) != nodes_per_step_target[n]:

                                                print('Fault in the replication process, the program is now closing')
                                                return 'Failure'

                                        ######################################################
                                        #               Graphic Section                      #
                                        ######################################################
                                        if (n >= self.starting_capture_time) & (n <= self.ending_capture_time):
                                                        if self.produce_videos_1 == 1:
                                                                    img       = np.reshape(grid/np.float(self.tau), (self.L,self.L))
                                                                            # Exctract its rgb version by using the dedicated function from opencv
                                                                    color_img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
                                                                    output    = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
                                                                    # Draw a red line representing the path of the heart beat
                                                                    if n == 1:
                                                                                cv2.line(color_img,((n)%self.T,0),((n)%self.T,self.L-1),(0,0,255))
                                                                    else:
                                                                                cv2.line(color_img,((n+1)%self.T,0),((n+1)%self.T,self.L-1),(0,0,255))
                                                                    # Draw the critical structures
                                                                    for path in cs_paths_cCMP:
                                                                                str_x    = np.array(path)%self.L
                                                                                str_y    = np.floor(np.array(path)/float(self.L))
                                                                                contours = [np.array([str_x[ij], str_y[ij]]) for ij in range(0,len(str_x),1)]
                                                                                ctr      = np.array(contours).reshape((-1,1,2)).astype(np.int32)
                                                                                cv2.drawContours(color_img,[ctr],-1,(255,0,0),-1)

                                                                    # Draw the dysfunctional cells
                                                                    for dysf_site in dysf_loc:
                                                                                d_x      = dysf_site%self.L
                                                                                d_y      = np.floor(dysf_site/float(self.L))
                                                                                contours = [np.array([d_x, d_y])]
                                                                                ctr      = np.array(contours).reshape((-1,1,2)).astype(np.int32)
                                                                                cv2.drawContours(color_img,[ctr],-1,(0,255,0),-1)

                                                                    cv2.addWeighted(color_img, alpha, output, 1 - alpha,0, output)
                                                                    output  = output[y_lims[0]:y_lims[1],x_lims[0]:x_lims[1]]
                                                                    output  = cv2.resize(output,(pic_width,pic_height))
                                                                    cv2.putText(output, "t={}".format(n), org=(10,20),fontFace=1, fontScale=1, color=(0,255,0), thickness=1)
                                                                    cv2.putText(output, "a={}".format(nodes_per_step_target[n]), org=(10,40),fontFace=1, fontScale=1, color=(0,255,0), thickness=1)
                                                                    out.write(output)

                                                        if self.produce_schemes_1 == 1:

                                                                    grid_container.append([list(np.repeat(np.maximum(el,0)/float(self.tau + 1),3)) for el in grid])




                                        print(n)
                                        if(n > self.ending_capture_time):
                                            break


                        if self.produce_videos_1 == 1:
                               cap.release()
                               out.release()
                               cv2.destroyAllWindows()
                        if self.produce_schemes_1 == 1:
                               ani = animation.FuncAnimation(fig, animate, frames= np.arange(0,len(grid_container),1), init_func=init, fargs=(scat, grid_container, self.T, self.L, np.arange(self.starting_capture_time, self.ending_capture_time + 1,1),ax,pml,),interval=self.interval, blit=False)
                               ani.save(scheme_directory + '\\' + schematic_name + '.mp4', fps=self.fps, dpi=250, bitrate=5000)

                               #fig, ax           = plt.subplots(len(target_frames), 1, figsize=(10, 1.5*len(target_frames)), dpi=300)
                               #figure_tag        = ['(a)','(b)','(c)','(d)','(e)']
                               #[mtd.plot_grid(ax[idx], figure_tag[idx], self.L, self.tau, self.T, vlinks_x, vlinks_y, dysf_loc, self.scheme_min_x, self.scheme_max_x, self.scheme_min_y, self.scheme_max_y, grid_container[target_frames[idx] - self.starting_capture_time], target_frames[idx]) for idx in range(0,len(target_frames),1)]
                               #plt.subplots_adjust(left=0.125,hspace = 1.0, top = 0.97, bottom = .06)
                               #plt.savefig(scheme_directory + '\\captions.png', dpi = 600)
                               #plt.close()



                        # Terminate the timer
                        end      = time.clock()

                        #output_1 = [af_risk, durations, cs_length, cs_paths, active_log, active_structures_timeline, af_risk_old, durations_old, monitor_matrix, transition_matrix]
                        print('Phase 1: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(self.nu, 2)))


        ################################################################################################################
        #                                                                                                              #
        #                                                Phase II                                                      #
        #                                                                                                              #
        ################################################################################################################

        if (self.produce_schemes_2 == 1) | (self.produce_videos_2 == 1):

                        # Load files
                        dysfunctional                 = np.load(CMP_folder + "experiment_" + str(simulation_id) + "\\dysfunctional_CMP_" + str(simulation_id) + '.npy')
                        down_links                    = np.load(CMP_folder + "experiment_" + str(simulation_id) + "\\down_links_" + str(simulation_id) + '.npy')
                        up_links                      = mtd.shift2(down_links, self.L)
                        nodes_per_step_target         = np.load(CMP_folder + "experiment_" + str(simulation_id) + "\\active_nodes_CMP_" + str(simulation_id) + '.npy')
                        cs_paths_CMP                  = np.load(CMP_folder + "experiment_" + str(simulation_id) + "\\cs_paths_CMP_" + str(simulation_id) + '.npy')
                        dysf_matrix_cmp               = (np.load(CMP_folder + "experiment_" + str(simulation_id) + "\\dysf_matrix_CMP_" + str(simulation_id) + '.npz'))['arr_0']
                        # the numpy array inputs are turned into 'C' array objects and conveniently renamed
                        # grid represents the flattened LxL grid
                        grid                          = np.zeros((self.L*self.L),dtype= TYPE1)
                        # Calculate the number of dysfunctional cells
                        n_dysf                        = np.count_nonzero(dysfunctional)
                        # Construct a (1 x n_dysf) array containing the indexes of the dysfunctional cells
                        dysf_loc                      = np.array(np.where(dysfunctional == 1)[0],dtype= TYPE1)
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
                        # Set the pacemaker type
                        pacemaker_cells               = left_edge
                        pacemaker_bool                = np.zeros((self.L*self.L), dtype= np.bool)
                        # Set the active cells array
                        active                        = (grid == self.tau + 1)
                        ################################################################################################################
                        #                                             Video Input                                                      #
                        ################################################################################################################
                        if (self.produce_videos_2 == 1):
                                    video_directory  = output_folder + "\\animations\\videos\\"+ "{:.2f}".format(self.nu)+"\\CMP\\experiment_" + str(simulation_id)
                                    os.makedirs(video_directory, exist_ok=True)
                                    cap    = cv2.VideoCapture(0)
                                    x_lims = [self.scheme_min_x, self.scheme_max_x]
                                    y_lims = [self.scheme_min_y, self.scheme_max_y]
                                    out    = cv2.VideoWriter(video_directory + "\\" + video_name + video_format, -1, fps, (pic_width,pic_height))
                                    #out    = cv2.VideoWriter(video_directory + "\\" + video_name + video_format,  cv2.VideoWriter_fourcc(*'X264'), fps, (pic_width,pic_height))


                        ################################################################################################################
                        #                                        Stylized Grid Initiation                                              #
                        ################################################################################################################
                        if (self.produce_schemes_2 == 1):

                                    # Set up the video writer and the target folder
                                    grid_container    = []
                                    scheme_directory  =  output_folder + "\\animations\\schematics\\"+ "{:.2f}".format(self.nu)+"\\CMP\\experiment_" + str(simulation_id)
                                    os.makedirs(scheme_directory, exist_ok=True)
                                    # Linearize vertical links
                                    vlinks_x          = (np.where(down_links == True)[0])%self.L
                                    vlinks_y          =  np.floor((np.where(down_links == True)[0])/float(self.L))
                                    # Open the figure object
                                    fig, ax           = plt.subplots(figsize=(15,7.5),dpi=200)
                                    # Initiate the grid
                                    scat, ax, pml     = mtd.initiate_grid(self.L,self.tau,self.T,vlinks_x,vlinks_y,dysf_loc,self.scheme_min_x, self.scheme_max_x, self.scheme_min_y, self.scheme_max_y, grid, ax)

                                    def init():
                                        return ax

                                    def animate(frame_id,scat,color_list,T,L,t_span,ax,pml):
                                        edge_colors = []
                                        for row in color_list[frame_id]:
                                            '''
                                            if row[0] == 1:
                                                edge_colors.append('blue')
                                            elif row[0] == 0:
                                                edge_colors.append('green')
                                            else:
                                                edge_colors.append('orange')
                                            '''
                                            edge_colors.append('black')
                                        scat.set_facecolors(color_list[frame_id])
                                        scat.set_edgecolors(edge_colors)
                                        pml.set_data([t_span[frame_id]%T,t_span[frame_id]%T],[0,L])
                                        #ax.set_title("t: " + str(t_span[frame_id]), fontsize = 18)
                                        # Case d
                                        if (t_span[frame_id]) >= 187289:
                                                rect = mpl.patches.Rectangle((13.5,L - 32 - 0.2), 35, 1.4, linewidth=2,edgecolor='blue',facecolor='none')
                                        else:
                                                rect = mpl.patches.Rectangle((13.5,L - 32 - 0.2), 35, 1.4, linewidth=2,edgecolor='grey',facecolor='none')
                                        if (t_span[frame_id]) >= 187356:
                                                rect2 = mpl.patches.Rectangle((8.5,L - 34 - 0.2), 34, 1.4, linewidth=2,edgecolor='blue',facecolor='none')
                                        else:
                                                rect2 = mpl.patches.Rectangle((8.5,L - 34 - 0.2), 34, 1.4, linewidth=2,edgecolor='grey',facecolor='none')
                                        '''
                                        # Case h
                                        if (t_span[frame_id]) >= 293734:
                                                rect = mpl.patches.Rectangle((149.5,L - 174 - 0.2), 29, 1.4, linewidth=2,edgecolor='blue',facecolor='none')
                                        else:
                                                rect = mpl.patches.Rectangle((149.5,L - 174 - 0.2), 29, 1.4, linewidth=2,edgecolor='grey',facecolor='none')
                                        '''
                                        ax.add_patch(rect)
                                        ax.add_patch(rect2)
                                        ax.set_title("t: " + str(t_span[frame_id]), fontsize = 18)

                                        print('Animation for timestep: ' + str(t_span[frame_id]) + ' completed.')
                                        return scat, pml,

                        ################################################################################################################
                        #                                             CMP Algorithm                                                    #
                        ################################################################################################################
                        # Initiate the timer
                        begin           = time.clock()
                        # Loop over the S steps, starting from step number 2 as the first one simply consists in the left edge pacemaker

                        for n in range(0,self.S,1):
                                        grid -= 1

                                        right_neighbors = mtd.shift2((active*right_links), 1)
                                        left_neighbors  = mtd.shift2((active*left_links), -1)
                                        down_neighbors  = mtd.shift2((active*down_links), self.L)
                                        up_neighbors    = mtd.shift2((active*up_links), -self.L)

                                        resting_cells   = grid < 1
                                        potential_sites = (right_neighbors) | (left_neighbors)| (down_neighbors) | (up_neighbors)

                                        temp_healthy            = np.copy(healthy_cells)
                                        temp_healthy[dysf_loc]  = dysf_matrix_cmp[n,:]



                                        temp_pacemaker_bool = np.copy(pacemaker_bool)
                                        temp_pacemaker_bool[pacemaker_cells] = (n % self.T == 0)*True

                                        active                = ((resting_cells)*(potential_sites)*(temp_healthy)) | ((temp_pacemaker_bool)*(resting_cells))
                                        grid[active]          = self.tau + 1

                                        if np.count_nonzero(active) != nodes_per_step_target[n]:
                                                print('Fault in the replication process, the program is now closing')
                                                return 'Failure'


                                        ######################################################
                                        #               Graphic Section                      #
                                        ######################################################
                                        if (n >= self.starting_capture_time) & (n <= self.ending_capture_time):
                                                        if self.produce_videos_2 == 1:
                                                                    img       = np.reshape(grid/np.float(self.tau), (self.L,self.L))
                                                                    # Exctract its rgb version by using the dedicated function from opencv
                                                                    color_img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
                                                                    output    = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
                                                                    # Draw a red line representing the path of the heart beat
                                                                    '''
                                                                    if n == 1:
                                                                                cv2.line(color_img,((n)%self.T,0),((n)%self.T,self.L-1),(0,0,255))
                                                                    else:
                                                                                cv2.line(color_img,((n+1)%self.T,0),((n+1)%self.T,self.L-1),(0,0,255))
                                                                    # Draw the critical structures
                                                                    
                                                                    for path in cs_paths_CMP:
                                                                                str_x    = np.array(path)%self.L
                                                                                str_y    = np.floor(np.array(path)/float(self.L))
                                                                                contours = [np.array([str_x[ij], str_y[ij]]) for ij in range(0,len(str_x),1)]
                                                                                ctr      = np.array(contours).reshape((-1,1,2)).astype(np.int32)
                                                                                cv2.drawContours(color_img,[ctr],-1,(255,0,0),-1)

                                                                    # Draw the dysfunctional cells
                                                                    for dysf_site in dysf_loc:
                                                                                d_x      = dysf_site%self.L
                                                                                d_y      = np.floor(dysf_site/float(self.L))
                                                                                contours = [np.array([d_x, d_y])]
                                                                                ctr      = np.array(contours).reshape((-1,1,2)).astype(np.int32)
                                                                                cv2.drawContours(color_img,[ctr],-1,(0,255,0),-1)
                                                                    '''
                                                                    cv2.addWeighted(color_img, alpha, output, 1 - alpha,0, output)
                                                                    output  = output[y_lims[0]:y_lims[1],x_lims[0]:x_lims[1]]
                                                                    output  = cv2.resize(output,(pic_width,pic_height))
                                                                    cv2.putText(output, "t={}".format(n), org=(10,20),fontFace=1, fontScale=1, color=(0,255,0), thickness=1)
                                                                    #cv2.putText(output, "a={}".format(nodes_per_step_target[n]), org=(10,40),fontFace=1, fontScale=1, color=(0,255,0), thickness=1)
                                                                    out.write(output)

                                                        if self.produce_schemes_2 == 1:
                                                                    grid_container.append([list(np.repeat(np.maximum(el,0)/float(self.tau + 1),3)) for el in grid])



                                        print(n)
                                        if(n > self.ending_capture_time):
                                            break




                        if self.produce_videos_2 == 1:
                                cap.release()
                                out.release()
                                cv2.destroyAllWindows()
                        if self.produce_schemes_2 == 1:
                                ani = animation.FuncAnimation(fig, animate, frames= np.arange(0,len(grid_container),1), init_func=init, fargs=(scat, grid_container, self.T, self.L, np.arange(self.starting_capture_time, self.ending_capture_time + 1,1),ax,pml,),interval=self.interval, blit=False)
                                ani.save(scheme_directory + '\\' + schematic_name + '.mp4', fps=self.fps, dpi=250, bitrate=5000)

                                #fig, ax           = plt.subplots(len(target_frames), 1, figsize=(10, 1.5*len(target_frames)), dpi=300)
                                #figure_tag        = ['(a)','(b)','(c)','(d)','(e)']
                                #[mtd.plot_grid(ax[idx], figure_tag[idx], self.L, self.tau, self.T, vlinks_x, vlinks_y, dysf_loc, self.scheme_min_x, self.scheme_max_x, self.scheme_min_y, self.scheme_max_y, grid_container[target_frames[idx] - self.starting_capture_time], target_frames[idx]) for idx in range(0,len(target_frames),1)]
                                #plt.subplots_adjust(left=0.125,hspace = 1.0, top = 0.97, bottom = .06)
                                #plt.savefig(scheme_directory + '\\captions.png', dpi = 600)
                                #plt.close()


                        end      = time.clock()
                        print('Phase 2: Processing time: ' + str(end - begin) +' secs., Simulation ID: ' + str(simulation_id + 1) + ', nu: ' + str(np.round(self.nu, 2)))


        return 'Success'





