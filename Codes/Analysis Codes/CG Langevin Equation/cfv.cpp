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
void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &vx, std::vector<double> &vy, std::vector<double> &vz, std::vector<double> &fx, std::vector<double> &fy, std::vector<double> &fz);
double signval(double x);
void wrap_coordinate(double &coordinate, const double box_length);
void min_image(double &displacement, const double box_length);


int main(int argc, char *argv[]) {
    std::cout << "Hello!" << std::endl;

    std::string traj_filename = argv[1];
    std::string output_filename = argv[2];

    // System specs
    size_t n_sites = 256;
    double box_length = 19.730000;

    // Histogram specs
    size_t n_time = 49998;
    size_t n_out= 1000;
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
    std::vector<double> vx(n_sites);
    std::vector<double> vy(n_sites);
    std::vector<double> vz(n_sites);
    std::vector<double> fx(n_sites);
    std::vector<double> fy(n_sites);
    std::vector<double> fz(n_sites);

// Maybe this needs to be extended to another vector (3D with 1000 X)
    std::vector<std::vector<std::vector<double> > > type2coord(1000, std::vector<std::vector <double> >(n_time, std::vector <double>(3, 0.0)));	// Type 2 coord -> Parse the type 2 one (not needed in here)
    std::vector<std::vector<std::vector<double> > > type2vel(1000, std::vector<std::vector <double> >(n_time, std::vector <double>(3, 0.0))); // Needs to calculate dx to "unwrap" the trajectory
    std::vector<std::vector<std::vector<double> > > type2force(1000, std::vector<std::vector <double> >(n_time, std::vector <double>(3, 0.0))); // Unwrapped trajectory -> This variable will be used for the MSD



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
            read_body_line(traj_filestream, i, type, x, y, z, vx, vy, vz, fx, fy, fz);
        }

        // Scale all coordinates by the box size and wrap.
        for (size_t i = 0; i < n_sites; ++i) {
            wrap_coordinate(x[i], box_length);
            wrap_coordinate(y[i], box_length);
            wrap_coordinate(z[i], box_length);
            type2coord[i][iframe][0] = x[i];
            type2coord[i][iframe][1] = y[i];
            type2coord[i][iframe][2] = z[i];
            type2vel[i][iframe][0] = vx[i];
            type2vel[i][iframe][1] = vy[i];
            type2vel[i][iframe][2] = vz[i];
            type2force[i][iframe][0] = fx[i];
            type2force[i][iframe][1] = fy[i];
            type2force[i][iframe][2] = fz[i];
        }
        skip_line(traj_filestream);
        iframe ++;
        if (iframe%1000 ==0){
        	// Print out the screen it every 1000 frames
            std::cout << iframe << " is processed!" << std::endl;
        }
    }

    for (size_t i=0; i<n_out; i=i+1){
    	double rmsd_value = 0.0;
    	size_t rmsd_index = 0;
    	for (size_t j=0; j<n_time-i;++j){ //j is the starting time frame 
		double rmsd_comp = 0.0;
    		for (size_t k=0; k<n_sites;++k){
            		double dx = type2force[k][i+j][0]*type2vel[k][j][0];
            		double dy = type2force[k][i+j][1]*type2vel[k][j][1];
            		double dz = type2force[k][i+j][2]*type2vel[k][j][2];
            		double distance_squared = dx+dy+dz;
			rmsd_comp += distance_squared;
            	}
		rmsd_value += rmsd_comp/((double)(n_sites));
		rmsd_index += 1;
        }
        histogram_vals[i] = rmsd_value/((double)(rmsd_index));
        if (i%100 == 0){
            std::cout << i << "/" << n_out << " is calculated!" << std::endl;
        }
    }

    // Print out the final g(r)
    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());
    for (size_t i = 0; i < n_out; i=i+1) {
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

void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &vx, std::vector<double> &vy, std::vector<double> &vz, std::vector<double> &fx, std::vector<double> &fy, std::vector<double> &fz) {
    std::string junk;
    // Skip ID.
    traj_filestream >> junk;
    // Read type.
    traj_filestream >> type[i];
    // Read (scaled) coordinate.
    traj_filestream >> x[i] >> y[i] >> z[i];
    // Skip force.
    traj_filestream >> vx[i] >> vy[i] >> vz[i];
    traj_filestream >> fx[i] >> fy[i] >> fz[i];
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

