#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

void skip_line(std::ifstream &traj_filestream);
int read_timestep(std::ifstream &traj_filestream);
void skip_natoms(std::ifstream &traj_filestream);
void skip_box(std::ifstream &traj_filestream);
void read_body_line(std::ifstream &traj_filestream, const int i, std::vector<int> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w);
double signval(double x);
void wrap_coordinate(double &coordinate, const double box_length);
void min_image(double &displacement, const double box_length);


int main(int argc, char *argv[]) {
    std::cout << "Hello!" << std::endl;

    std::string traj_filename = argv[1];
    std::string output_filename = argv[2];

    // System specs
    int n_sites = 256;
    double box_length = 19.73;
    std::string evb_name = argv[3];
    int n_hyd = 2;
    // Histogram specs
    int n_time = 100000;
    int n_out= 10000;
    std::vector<double> histogram_vals(n_out, 0.0);
    std::vector<double> histogram_vals2(n_out, 0.0);
    std::vector<double> histogram_valst(n_out, 0.0);

    std::ifstream traj_filestream;
    traj_filestream.open(traj_filename.c_str());
    std::string junk;
    int curr_timestep = 0;
    std::vector<int> type(n_sites);
    std::vector<double> x(n_sites);
    std::vector<double> y(n_sites);
    std::vector<double> z(n_sites);
    std::vector<double> w(n_sites);
 //   std::vector<int> l2list(n_hyd);
  //  std::vector<int> k2list(n_hyd);
    std::vector<std::vector <int> > l2list(n_time, std::vector<int>(n_hyd, -5));
    std::vector<std::vector <int> > k2list(n_time, std::vector<int>(n_hyd, 100)); // Used to swap the id
    std::vector<int> prevlist(n_hyd);
    std::vector<std::vector <std::vector <double> > >type2coord(n_time, std::vector<std::vector <double> >(n_hyd, std::vector<double>(3, 0.0)));
    std::vector<std::vector <std::vector <double> > >type2coord_tmp(n_time, std::vector<std::vector <double> >(n_hyd, std::vector<double>(3, 0.0))); // Used to swap the id
    std::vector<std::vector <std::vector <double> > >type2_diff(n_time-1, std::vector<std::vector <double> >(n_hyd, std::vector<double>(3, 0.0)));
    std::vector<std::vector <std::vector <double> > > type2_fin(n_time, std::vector<std::vector <double> >(n_hyd, std::vector<double>(3, 0.0)));
    std::vector<std::vector <int> > type2idx(n_time, std::vector<int>(n_hyd, 0));
    std::vector<std::vector <int> > type2idx_tmp(n_time, std::vector<int>(n_hyd, 0)); // Used to swap the id
//    std::vector<std::vector <double> > type2coord(n_time, std::vector<double>(3, 0.0));
//    std::vector<std::vector <double> > type2_diff(n_time-1, std::vector<double>(3, 0.0));
//    std::vector<std::vector <double> > type2_fin(n_time, std::vector<double>(3, 0.0));


    // For each frame in the trajectory
    int iframe = 0;
    int checkval;
    int checkval_k;
    while (iframe<n_time) {
        // Read header lines.
        // Read timestep.
        curr_timestep = read_timestep(traj_filestream);
        // Skip the number of atoms.
        skip_natoms(traj_filestream);
        // Skip reading box dimensions.
        skip_box(traj_filestream);
        // Skip the final (body) header line.
        skip_line(traj_filestream);

        // Read body lines.
        for (int i = 0; i < n_sites; ++i) {
            read_body_line(traj_filestream, i, type, x, y, z, w);
        }

        // Scale all coordinates by the box size and wrap.
        int hyd_count = 0;
        for (int i = 0; i < n_sites; ++i) {
            if(type[i] == 2){
                wrap_coordinate(x[i], box_length);
                wrap_coordinate(y[i], box_length);
                wrap_coordinate(z[i], box_length);
                if (iframe == 0){
                    type2_fin[0][hyd_count][0] = x[i];
                    type2_fin[0][hyd_count][1] = y[i];
                    type2_fin[0][hyd_count][2] = z[i];                    
                    type2coord[0][hyd_count][0] = x[i];
                    type2coord[0][hyd_count][1] = y[i];
                    type2coord[0][hyd_count][2] = z[i];                    
                    type2idx[0][hyd_count] = i;                    
                    type2idx_tmp[0][hyd_count] = i;                    
                    hyd_count += 1;
                }
                if (iframe != 0){
                    type2coord_tmp[iframe][hyd_count][0] = x[i];
                    type2coord_tmp[iframe][hyd_count][1] = y[i];
                    type2coord_tmp[iframe][hyd_count][2] = z[i];
                    type2idx_tmp[iframe][hyd_count] = i;                    
                    hyd_count += 1;
                }
            }
        }
        if (iframe != 0){
            int same_hyd = 0;
            for (int k=0;k<n_hyd;++k){ // previous
                for (int k1=0;k1<n_hyd;++k1){ // new
                    if (type2idx[iframe-1][k] == type2idx_tmp[iframe][k1]){
                        l2list[iframe][k]=type2idx_tmp[iframe][k1];
                        k2list[iframe][k]=k1;
                        same_hyd +=1;
                    }
                }
            }
            if (same_hyd != n_hyd){
                for (int k=0;k<n_hyd;++k){  //prev
                    int checkval_k = 1;
                    for (int k2 = 0;k2<n_hyd;++k2){ //check if it is in l2list to sort out proper (k,k1) pair
                        checkval_k = checkval_k*(type2idx[iframe-1][k]-l2list[iframe][k2]);
                    }
                    if (checkval_k == 0){
                        continue;
                    }
                    else{
                        double max_value = 500000.0;
                        int max_index = 0;
                        int k_index = -1;
                        for (int k1 = 0;k1<n_hyd;++k1){ //new
                            int checkval = 1;
                            for (int k2=0;k2<n_hyd;++k2){ //check if it is in l2list to sort out proper (k,k1) pair
                                checkval = checkval*(type2idx_tmp[iframe][k1]-l2list[iframe][k2]);
                            }
                            if (checkval != 0){
                                double dx = type2coord_tmp[iframe][k1][0]-type2coord[iframe-1][k][0];
                                double dy = type2coord_tmp[iframe][k1][1]-type2coord[iframe-1][k][1];
                                double dz = type2coord_tmp[iframe][k1][2]-type2coord[iframe-1][k][2];
                                double distance_squared = dx * dx + dy * dy + dz * dz;
                                double distance = sqrt(distance_squared);
                                if (distance >= box_length/2.0) {
                                    if (sqrt(dx*dx) >= box_length/2.0){
                                            dx = dx - signval(dx)*box_length;
                                    }
                                    if (sqrt(dy*dy) >= box_length/2.0){
                                            dy = dy - signval(dy)*box_length;
                                    }
                                    if (sqrt(dz*dz) >= box_length/2.0){
                                            dz = dz - signval(dz)*box_length;
                                    }
                                }  
                                distance_squared = dx * dx + dy * dy + dz * dz;
                                if (distance_squared < max_value){
                                    max_value = distance_squared;
                                    max_index = type2idx_tmp[iframe][k1];
                                    k_index = k1;
                                }
                            }
                        }
                        l2list[iframe][k] = max_index;
                        k2list[iframe][k] = k_index;
                    }
                }
            }
            for (int k=0;k<n_hyd;++k){
                //std::cout << iframe << " " << k2list[iframe][k] << " " << l2list[iframe][k] << std::endl;
                type2idx[iframe][k] = l2list[iframe][k];
                type2coord[iframe][k][0]=type2coord_tmp[iframe][k2list[iframe][k]][0];
                type2coord[iframe][k][1]=type2coord_tmp[iframe][k2list[iframe][k]][1];
                type2coord[iframe][k][2]=type2coord_tmp[iframe][k2list[iframe][k]][2];
            }
            for (int k=0;k<n_hyd;++k){
                double dx = type2coord[iframe][k][0] - type2coord[iframe-1][k][0];
                double dy = type2coord[iframe][k][1] - type2coord[iframe-1][k][1];
                double dz = type2coord[iframe][k][2] - type2coord[iframe-1][k][2];
                double distance_squared = dx * dx + dy * dy + dz * dz;
                double distance = sqrt(distance_squared);
                if (distance >= box_length/2.0) {
                    if (sqrt(dx*dx) >= box_length/2.0){
                    	dx = dx - signval(dx)*box_length;
                    }
                    if (sqrt(dy*dy) >= box_length/2.0){
                    	dy = dy - signval(dy)*box_length;
                    }
                    if (sqrt(dz*dz) >= box_length/2.0){
                    	dz = dz - signval(dz)*box_length;
                    }
                }
                type2_diff[iframe-1][k][0] = dx;
                type2_diff[iframe-1][k][1] = dy;
                type2_diff[iframe-1][k][2] = dz;
                type2_fin[iframe][k][0] = type2_fin[iframe-1][k][0]+dx;
                type2_fin[iframe][k][1] = type2_fin[iframe-1][k][1]+dy;
                type2_fin[iframe][k][2] = type2_fin[iframe-1][k][2]+dz;                     
            }
        }
        skip_line(traj_filestream);
        iframe ++;
        if (iframe%1000 ==0){
            std::cout << iframe << " is processed!" << std::endl;
        }
    }

    for (int i=10; i<n_out; i=i+10){
    	double rmsd_value = 0.0;
    	int rmsd_index = 0;
    	double rmsd_value2 = 0.0;
    	int rmsd_index2 = 0;
    	double rmsd_valuet = 0.0;
    	int rmsd_indext = 0;
    	for (int j=0; j<n_time-i;++j){
            for (int k=0;k<n_hyd;++k){
                double dx = type2_fin[i+j][k][0] - type2_fin[j][k][0];
                double dy = type2_fin[i+j][k][1] - type2_fin[j][k][1];
                double dz = type2_fin[i+j][k][2] - type2_fin[j][k][2];
                double distance_squared = dx * dx + dy * dy + dz * dz;
                if (k == 0){
                    rmsd_index += 1;
                    rmsd_value += distance_squared;
                }
                else{
                    rmsd_index2 += 1;
                    rmsd_value2 += distance_squared;
                }
                rmsd_indext += 1;
                rmsd_valuet += distance_squared;
            }
        }
        histogram_vals[i] = rmsd_value/((double)(rmsd_index));
        histogram_vals2[i] = rmsd_value2/((double)(rmsd_index2));
        histogram_valst[i] = rmsd_valuet/((double)(rmsd_indext));
        if (i%100 == 0){
            std::cout << i << "/" << n_out << " is calculated!" << std::endl;
        }
    }

    // Print out the final g(r)
    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());
    for (int i = 10; i < n_out; i=i+10) {
        if (histogram_vals[i]*histogram_vals2[i]*histogram_valst[i]!=0){
            output_filestream << i << " " << histogram_vals[i] << " " << histogram_vals2[i] << " " << histogram_valst[i] << std::endl;
        }
    }

    return EXIT_SUCCESS;
}

void skip_line(std::ifstream &traj_filestream) {
    std::string junk;
    std::getline(traj_filestream, junk);
}

int read_timestep(std::ifstream &traj_filestream) {
    int timestep;
    skip_line(traj_filestream);
    traj_filestream >> timestep;
    skip_line(traj_filestream);
    return timestep;
}

void skip_natoms(std::ifstream &traj_filestream) {
    skip_line(traj_filestream);
    skip_line(traj_filestream);
}

void skip_box(std::ifstream &traj_filestream) {
    skip_line(traj_filestream);
    skip_line(traj_filestream);
    skip_line(traj_filestream);
    skip_line(traj_filestream);
}

void read_body_line(std::ifstream &traj_filestream, const int i, std::vector<int> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w) {
    std::string junk;
    // Skip ID.
    traj_filestream >> junk;
    // Read type.
    traj_filestream >> type[i];
    // Read (scaled) coordinate.
    traj_filestream >> x[i] >> y[i] >> z[i];
    // Skip force.
    traj_filestream >> junk  >> junk  >> junk;
}

void wrap_coordinate(double &coordinate, const double box_length) {
    while (coordinate < 0.0) {
        coordinate += box_length;
    }
    while (coordinate >= box_length) {
        coordinate -= box_length;
    }
}

double signval(double x) {
  if (x > 0.0) return 1.0;
  if (x < 0.0) return -1.0;
  return 0;
}

void min_image(double &displacement, const double box_length) {
    displacement -= box_length * round(displacement / box_length);
}

