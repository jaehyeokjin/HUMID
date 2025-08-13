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
    size_t n_time = 1000000;
    size_t n_out= 1000000;
    std::vector<size_t> histogram_vals(n_out, 0.0);

    std::ifstream traj_filestream;
    traj_filestream.open(traj_filename.c_str());
    std::string junk;
    size_t curr_timestep = 0;

    std::vector<size_t> type(n_sites);
    std::vector<double> x(n_sites);
    std::vector<double> y(n_sites);
    std::vector<double> z(n_sites);
    std::vector<double> w(n_sites);

    std::vector<size_t> type2id(n_time);

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
            if(type[i] == 2){
                type2id[iframe] = i;
                break;
        	}
        }
        skip_line(traj_filestream);
        iframe ++;
        if (iframe%1000 ==0){
            std::cout << iframe << " is processed!" << std::endl;
        }
    }
    // Print out the final g(r)
    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());
    for (size_t i = 0; i < n_out; ++i) {
        output_filestream << i << " " << type2id[i] << std::endl;
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

