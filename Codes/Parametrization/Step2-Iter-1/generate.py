import copy
import math
import sys
import numpy as np
import random

# Initial variables
outputname = "ucg.lammpstrj"
inputname = sys.argv[1]
f = open(inputname, 'r')
out_data = open(outputname, 'w')
k = 0
count_time = 0
index = [i for i in range(0, 256)]
index_temp = []
coord_meoh= []
force = []
time_element = 0
# UCG treatment
r_th = 2.5
wcut = 0.85
cut_off = 0.15
box_size = 19.73
n_traj = 20
# Rand function

# Pre-defined string (in order to save the time for processing)
str_a = "ITEM: TIMESTEP\n"
str_b = "\nITEM: NUMBER OF ATOMS\n256\nITEM: BOX BOUNDS pp pp pp\n0.00000 19.73\n0.00000 19.73\n0.00000 19.73\nITEM: ATOMS id type x y z fx fy fz\n"
###########################################################
############### Trajectory Processing ...  ################
###########################################################
time_frame=0
for line in f:
    line_element = line.split()
    if(line_element[0] == 'ITEM:'):
        k = (k+1) % 4
    if(k == 1 and len(line_element) == 1):
        time_element = int(line_element[0])
        index_temp = []
        coord_meoh= []
        force = []
    if(k == 0 and line_element[0] != 'ITEM:'):  # Scan for all
        index_temp.append(int(line_element[0])) # Until the one time frame is successfully parsed
        coord_meoh.append([float(line_element[2]), float(line_element[3]), float(line_element[4])])
        force.append([float(line_element[5]), float(line_element[6]), float(line_element[7])]) # Force is not changed
        if (len(index_temp) == 256):  # After parsing, construct the molecule-wise distance
            w = [0.0 for _ in range(0,256)]
            for i in range(0, 256):
                w_value = 0.0
                for j in range(0, 256):
                    if i!= j:
                        delta = [coord_meoh[i][0] - coord_meoh[j][0], coord_meoh[i][1] - coord_meoh[j][1], coord_meoh[i][2] - coord_meoh[j][2]]
                        for ii in range(0, 3):
                            while(delta[ii] > box_size/2.0 or delta[ii] < -0.5 * box_size):
                                if delta[ii] > box_size/2.0:
                                    delta[ii] -= box_size
                                elif delta[ii] < -0.5 * box_size:
                                    delta[ii] += box_size
                        distance = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]) ** 0.5 # Update the distance
                        w_value += 1.0/(1.0+math.exp(11.0*(distance-2.5)))
                w[i] = w_value
            local_type = [[1 for _ in range(0,n_traj)] for _ in range(0,256)]
            local_prob = [0.0 for _ in range(0,256)]
            for i in range(0, 256):
                local_prob[i] =  0.5 * (1.0 + math.tanh((w[i] - wcut) / (cut_off * wcut)))
            prob_sort = sorted(local_prob)
            for i in range (0,10): # Number of 2 States in the system
                hyd_id = local_prob.index(prob_sort[len(prob_sort)-i-1])
                prob_value = int(round(local_prob[hyd_id]*float(n_traj)))
                for kk in range(0,n_traj):
                    if (prob_value >= 0):
                        local_type[hyd_id][kk] = 2
                        prob_value -= 1
                    else:
                        local_type[hyd_id][kk] = 1
                random.shuffle(local_type[hyd_id])

            for iter_num in range(0,n_traj):
                str_header = str_a + str(time_frame) + str_b
                out_data.write(str_header)
                for i in range(0, 256):
                    line_methanol = str(i+1) + ' ' + str(local_type[i][iter_num]) + ' %.6f %.6f %.6f %2.10f %2.10f %2.10f\n' % (coord_meoh[i][0], coord_meoh[i][1], coord_meoh[i][2], force[i][0], force[i][1], force[i][2])
                    out_data.write(line_methanol)
                time_frame += 1
            # Clean up the variables
            index_temp= []
            coord_meoh= []
            print_arg = "Time step: %d is done \n" % (time_element)
            print(print_arg)

f.close()
out_data.close()

