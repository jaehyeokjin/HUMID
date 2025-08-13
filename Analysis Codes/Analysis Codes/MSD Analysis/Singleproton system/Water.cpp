#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

void skip_line(std::ifstream &traj_filestream);
size_t read_timestep(std::ifstream &traj_filestream);
void skip_natoms(std::ifstream &traj_filestream);
void skip_box(std::ifstream &traj_filestream);
void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w);
double signval(double x);
void wrap_coordinate(double &coordinate, const double box_length);
void min_image(double &displacement, const double box_length);


int main(int argc, char *argv[]) {
    std::cout << "Hello!" << std::endl;

    std::string traj_filename = argv[1];
    std::string output_filename = argv[2];

    // System specs
    size_t n_sites = 256;
    double box_length = 19.73;

    // Histogram specs
    size_t n_time = 100000;
    size_t n_out= 10000;
    std::vector<double> histogram_vals(n_out, 0.0);

    std::ifstream traj_filestream;
    traj_filestream.open(traj_filename.c_str());
    std::string junk;
    size_t curr_timestep = 0;

    std::vector<size_t> type(n_sites);
    std::vector<double> x(n_sites);
    std::vector<double> y(n_sites);
    std::vector<double> z(n_sites);
    std::vector<double> w(n_sites);

// Maybe this needs to be extended to another vector (3D with 256 X)
    std::vector<std::vector<std::vector<double> > > type2coord(256, std::vector<std::vector <double> >(n_time, std::vector <double>(3, 0.0)));	// Type 2 coord -> Parse the type 2 one (not needed in here)
    std::vector<std::vector<std::vector<double> > > type2_diff(256, std::vector<std::vector <double> >(n_time-1, std::vector <double>(3, 0.0))); // Needs to calculate dx to "unwrap" the trajectory
    std::vector<std::vector<std::vector<double> > > type2_fin(256, std::vector<std::vector <double> >(n_time, std::vector <double>(3, 0.0))); // Unwrapped trajectory -> This variable will be used for the MSD



    // For each frame in the trajectory
    size_t iframe = 0;
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
        for (size_t i = 0; i < n_sites; ++i) {
            read_body_line(traj_filestream, i, type, x, y, z, w);
        }

        // Scale all coordinates by the box size and wrap.
        for (size_t i = 0; i < n_sites; ++i) {
            wrap_coordinate(x[i], box_length);
            wrap_coordinate(y[i], box_length);
            wrap_coordinate(z[i], box_length);
            if (iframe == 0){
            	// Initial conditions
                type2_fin[i][0][0] = x[i];
                type2_fin[i][0][1] = y[i];
                type2_fin[i][0][2] = z[i];                    
            }
            if (iframe != 0){
                double dx = x[i] - type2coord[i][iframe-1][0];
                double dy = y[i] - type2coord[i][iframe-1][1];
                double dz = z[i] - type2coord[i][iframe-1][2];
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
                type2_diff[i][iframe-1][0] = dx;
                type2_diff[i][iframe-1][1] = dy;
                type2_diff[i][iframe-1][2] = dz;
                type2_fin[i][iframe][0] = type2_fin[i][iframe-1][0]+dx;
                type2_fin[i][iframe][1] = type2_fin[i][iframe-1][1]+dy;
                type2_fin[i][iframe][2] = type2_fin[i][iframe-1][2]+dz;                     
            }
            type2coord[i][iframe][0] = x[i];
            type2coord[i][iframe][1] = y[i];
            type2coord[i][iframe][2] = z[i];
        }
        skip_line(traj_filestream);
        iframe ++;
        if (iframe%1000 ==0){
        	// Print out the screen it every 1000 frames
            std::cout << iframe << " is processed!" << std::endl;
        }
    }

    for (size_t i=99; i<n_out; i=i+100){
    	double rmsd_value = 0.0;
    	size_t rmsd_index = 0;
    	for (size_t j=0; j<n_time-i;++j){ //j is the starting time frame 
    		for (size_t k=0; k<n_sites;++k){
            		double dx = type2_fin[k][i+j][0] - type2_fin[k][j][0];
            		double dy = type2_fin[k][i+j][1] - type2_fin[k][j][1];
            		double dz = type2_fin[k][i+j][2] - type2_fin[k][j][2];
            		double distance_squared = dx * dx + dy * dy + dz * dz;
            		rmsd_index += 1;
            		rmsd_value += distance_squared;
            	}
        }
        histogram_vals[i] = rmsd_value/((double)(rmsd_index));
    }

    // Print out the final g(r)
    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());
    for (size_t i = 0; i < n_out; ++i) {
        if (histogram_vals[i] !=0){
            output_filestream << i << " " << histogram_vals[i] << std::endl;
        }
    }

    return EXIT_SUCCESS;
}

void skip_line(std::ifstream &traj_filestream) {
    std::string junk;
    std::getline(traj_filestream, junk);
}

size_t read_timestep(std::ifstream &traj_filestream) {
    size_t timestep;
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

void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w) {
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

