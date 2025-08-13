import copy
import math
import sys
import numpy as np

# Initial variables
filename = sys.argv[1]
meoh_number = 1224
meoh_atom = 1224 # Neopentane is here
water_number = 48
water_atom = 3* water_number
total_number = water_atom + meoh_atom
header = filename.split(".")
outputname = "cg.lammpstrj"
f = open(filename, 'r')
out_data = open(outputname, 'w')
k = 0
count_time = 0
index = [i for i in range(0, total_number)]
index_temp = []
coord = []
force = []
time_element = 0

# PBC consideration
xlo = xhi = ylo = yhi = zlo = zhi = box_size = 0.0
slab_size = 0.0
axis_read = 0
# Wrapping function

def scale_position(pos, lo, hi):
    scaled_pos = pos * (hi-lo)
    return(scaled_pos)

def scale_slab(pos_s, lo_s, hi_s):
    scaled_slab = pos_s * (hi_s - lo_s)
    return(scaled_slab)

def distance(a, b):
    dist = ((a[0]-b[0])**2.0 + (a[1]-b[1])**2.0 + (a[2]-b[2])**2.0)**0.5
    return(dist)
###########################################################
############### Trajectory Processing ...  ################
###########################################################

for line in f:
    line_element = line.split()
    if(line_element[0] == 'ITEM:'):
        k = (k+1) % 4
        if (len(line_element) != 13):
            out_data.write(line)
        else:
            no_velocity = 'ITEM: ATOMS id type x y z fx fy fz\n'
            out_data.write(no_velocity)
    if(k == 1 and len(line_element) == 1):
        time_element = int(line_element[0])
        index_temp = []
        coord = []
        force = []
        out_data.write(line)
    if(k == 2 and len(line_element) == 1):
        n_mole = "1272\n"
        out_data.write(n_mole)
    if(k == 3 and len(line_element) == 2):
        if (axis_read == 0):
            xlo = float(line_element[0])
            xhi = float(line_element[1])
            axis_read += 1
            box_size = xhi-xlo
            pbc_str = "0.00000" + ' %2.5f\n' % (box_size)
            out_data.write(pbc_str)
        elif (axis_read == 1):
            ylo = float(line_element[0])
            yhi = float(line_element[1])
            axis_read += 1
            pbc_str = "0.00000" + ' %2.5f\n' % (box_size)
            out_data.write(pbc_str)
        elif (axis_read == 2):
            zlo = float(line_element[0])
            zhi = float(line_element[1])
            slab_size = zhi-zlo
            axis_read = 0
            pbc_str = "0.00000" + ' %2.5f\n' % (slab_size)
            out_data.write(pbc_str)
    if(k == 0 and line_element[0] != 'ITEM:'):  # Scan for all
        coord.append([float(line_element[2]), float(line_element[3]), float(line_element[4])])
        force.append([float(line_element[8]), float(line_element[9]), float(line_element[10])])
        index_temp.append(int(line_element[0]))
        if (len(index_temp) == total_number):  # Start the mapping process
            # Initialization
            com_methanol = []
            com_neopentane = []
            com_methanol_force = []
            com_neopentane_force = []
            # Sorting by the atom id
            size_case = len(coord)
            print_arg = "Time step: %d is starting! with %d %f %f %f \n" % (time_element, size_case, xlo, ylo, zlo)
            print(print_arg)
            # Molecule-wise COM
            # First case: methanol (1~meoh_number atoms i.e. 1~1000 molecules)
            for i in range(0, meoh_number):
                methanol = []
                methanol_force = []
                # Append to the methanol array
                temp = [scale_position(coord[i][0], xlo, xhi), scale_position(coord[i][1], ylo, yhi), scale_position(coord[i][2], zlo, zhi)]
                temp_force = [force[i][0], force[i][1], force[i][2]]
                # Vectorization (fix the first particle as the origin)
                for jj in range(0, 2):
                    while (temp[jj] > box_size or temp[jj] < 0.0):
                        if temp[jj] > box_size:
                            temp[jj] -= box_size
                        elif temp[jj] < 0.0:
                            temp[jj] += box_size
                while(temp[2] > slab_size/2.0 or temp[2] < -0.5*slab_size):
                    if temp[2] > slab_size/2.0:
                        temp[2] -= slab_size
                    elif temp[2] < -0.5*slab_size:
                        temp[2] += slab_size
                # PBC wrap
                com_methanol.append(temp)
                com_methanol_force.append(temp_force)
            # Molecule-wise COM
            # Second case: neopentane (6001~6850 atoms i.e. 1~50 molecules)
            for i in range(0, water_number):
                neopentane = []
                neopentane_force = []
                # Append to the neopentane array
                for j in range(0, 3):
                    temp = [scale_position(coord[3*i+j+meoh_atom][0], xlo, xhi), scale_position(coord[3*i+j+meoh_atom][1], ylo, yhi), scale_position(coord[3*i+j+meoh_atom][2], zlo, zhi)]
                    neopentane.append(temp)
                    temp_force = [force[3*i+j+meoh_atom][0], force[3*i+j+meoh_atom][1], force[3*i+j+meoh_atom][2]]
                    neopentane_force.append(temp_force)
                # Vectorization (fix the first particle as the origin)
                dx = []
                for j in range(1, 3):
                    delta = [neopentane[j][0]-neopentane[0][0], neopentane[j][1]-neopentane[0][1], neopentane[j][2]-neopentane[0][2]]
                    for ii in range(0, 2):
                        while(delta[ii] > box_size/2.0 or delta[ii] < -0.5 * box_size):
                            if delta[ii] > box_size/2.0:
                                delta[ii] -= box_size
                            elif delta[ii] < -0.5 * box_size:
                                delta[ii] += box_size
                    while(delta[2] > slab_size/2.0 or delta[2] < -0.5*slab_size):
                        if delta[2] > slab_size/2.0:
                            delta[2] -= slab_size
                        elif delta[2] < -0.5*slab_size:
                            delta[2] += slab_size
                    dx.append(delta)
                # COM calculation
                neopentane_unwrap = []
                neopentane_unwrap.append(neopentane[0])
                for j in range(1, 3):
                    unwrap_temp = [neopentane[0][0] + dx[j-1][0],neopentane[0][1] + dx[j-1][1],neopentane[0][2] + dx[j-1][2]]
                    neopentane_unwrap.append(unwrap_temp)
                com_final = (15.999000 * (np.array(neopentane_unwrap[2])) + 1.008000*(np.array(neopentane_unwrap[1])+np.array(neopentane_unwrap[0])))/18.01500
                com_neo = (np.array(neopentane_force[0])+np.array(neopentane_force[1])+np.array(neopentane_force[2]))
                for jj in range(0, 2):
                    while (com_final[jj] > box_size or com_final[jj] < 0.0):
                        if com_final[jj] > box_size:
                            com_final[jj] -= box_size
                        elif com_final[jj] < 0.0:
                            com_final[jj] += box_size
                while (com_final[2] > slab_size or com_final[2] < 0.0):
                    if com_final[2] > slab_size:
                        com_final[2] -= slab_size
                    elif com_final[2] < 0.0:
                        com_final[2] += slab_size
                com_neopentane.append(com_final)
                com_neopentane_force.append(com_neo)
            # Print out the CG-ed sites
            for i in range(0, meoh_number):
                methanol_output = str(i+1) + ' 1 %.6f %.6f %.6f %2.10f %2.10f %2.10f\n' % (com_methanol[i][0], com_methanol[i][1], com_methanol[i][2], com_methanol_force[i][0], com_methanol_force[i][1], com_methanol_force[i][2])
                out_data.write(methanol_output)
            for i in range(0, water_number):
                neopentane_output = str(i+1+meoh_number) + ' 2 %.6f %.6f %.6f %2.10f %2.10f %2.10f\n' % (com_neopentane[i][0], com_neopentane[i][1], com_neopentane[i][2], com_neopentane_force[i][0], com_neopentane_force[i][1], com_neopentane_force[i][2])
                out_data.write(neopentane_output)
            # Clean up the variables
            coord = []
            index_temp = []
            print_arg = "Time step: %d is done! \n" % time_element
            print(print_arg)

f.close()
out_data.close()



