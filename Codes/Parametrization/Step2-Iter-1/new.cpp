#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iomanip> 
#include <random>
#include <iterator>
#include <algorithm>

void skip_line(std::ifstream &traj_filestream);
size_t read_timestep(std::ifstream &traj_filestream);
void skip_natoms(std::ifstream &traj_filestream);
void skip_box(std::ifstream &traj_filestream);
void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &fx, std::vector<double> &fy, std::vector<double> &fz);

void wrap_coordinate(double &coordinate, const double box_length);
void wrap_coordinate_slab(double &coordinate, const double slab_length);
void min_image(double &displacement, const double box_length);
void min_image_slab(double &displacement, const double slab_length);


int main(int argc, char *argv[]) {
    std::cout << "Hello!" << std::endl;

    std::string traj_filename = argv[1];
    std::string output_filename = argv[2];

    // System specs
    size_t n_sites = 256;
    size_t cg_site = 256;
    size_t n_traj = 50;
    double box_length = 19.73;
    double slab_length = 19.73;

    // Histogram specs
    double w_th = 1.05; 
    double random_value = 0.0;

    std::vector<double> probability_22(n_sites, 0.0);
    std::ifstream traj_filestream;
    traj_filestream.open(traj_filename.c_str());

    std::string junk;
    size_t curr_timestep = 0;

    std::vector<size_t> type(n_sites);
    std::vector<double> x(n_sites);
    std::vector<double> y(n_sites);
    std::vector<double> z(n_sites);
    std::vector<double> xu(n_sites);
    std::vector<double> yu(n_sites);
    std::vector<double> zu(n_sites);
    std::vector<double> fx(n_sites);
    std::vector<double> fy(n_sites);
    std::vector<double> fz(n_sites);

    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());

    // For each frame in the trajectory
    size_t iframe = 0;
    size_t time_index = 0;
    while ((traj_filestream.good())) {
        // Read header lines.
        // Read timestep.
        curr_timestep = read_timestep(traj_filestream);
        // Skip the number of atoms.
        skip_natoms(traj_filestream);
        // Skip reading box dimensions.
        skip_box(traj_filestream);
        std::cout << curr_timestep << std::endl;
        // Skip the final (body) header line.
        skip_line(traj_filestream);

        // Read body lines.
        for (size_t i = 0; i < n_sites; ++i) {
            read_body_line(traj_filestream, i, type, x, y, z, fx, fy, fz);
            xu[i] = x[i];
            yu[i] = y[i];
            zu[i] = z[i];
        }

        // Scale all coordinates by the box size and wrap.
        for (size_t i = 0; i < n_sites; ++i) {
            wrap_coordinate(x[i], box_length);
            wrap_coordinate(y[i], box_length);
            wrap_coordinate_slab(z[i], slab_length);
        }

        skip_line(traj_filestream);

        // Do distance analysis.
        // Populate the radial histogram.
        std::vector<std::vector<int>> local_type_22(cg_site, std::vector<int>(n_traj, 0));
        for (size_t i = 0; i < cg_site; ++i) {
            double local_density_22 = 0.0;
            for (size_t j = 0; j < cg_site; ++j) {
                if (i != j)  {
                    double dx22 = x[i] - x[j];
                    double dy22 = y[i] - y[j];
                    double dz22 = z[i] - z[j];
                    min_image(dx22, box_length);
                    min_image(dy22, box_length);
                    min_image_slab(dz22, slab_length);
                    double distance_squared_22 = dx22 * dx22 + dy22 * dy22 + dz22 * dz22;
                    // Accumulate in histogram if it belongs.
                    local_density_22 += 1.0/(1.0+exp(11.0*(sqrt(distance_squared_22)-2.5)));
                    // Calculate the distance squared.
                }
            }
            probability_22[i] = 0.5*(1.0+tanh((local_density_22-w_th)/(0.25*w_th)));
            int prob_2 = ((int) round(probability_22[i]*50.0));
	    for (size_t k = 0; k < n_traj; ++k){
                if( prob_2 >= 0){
                    local_type_22[i][k] = 2;
                    prob_2 -= 1;
                }
                else{
                    local_type_22[i][k] = 1;
                }
            }
	    std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(local_type_22[i].begin(), local_type_22[i].end(), g);
        }
        for (size_t i = 0; i < n_traj; ++i){
	    output_filestream << "ITEM: TIMESTEP\n" << time_index << "\nITEM: NUMBER OF ATOMS\n256\nITEM: BOX BOUNDS pp pp pp\n0.00000 19.73000\n0.00000 19.73000\n0.00000 19.73000\nITEM: ATOMS id type x y z fx fy fz"  << std::endl;
            for (size_t j = 0; j < cg_site; ++j){
                size_t index_22 = j+1;
                output_filestream << index_22 << " " << local_type_22[j][i] << " " << std::setprecision(6) << std::fixed << xu[j] << " " << yu[j] << " " << zu[j] << " " << std::setprecision(10) << std::fixed << fx[j]<< " " << fy[j]<< " " << fz[j] << std::endl;
            }
            time_index += 1;
        }
        iframe++;
        std::cout << iframe << " is processed!" << std::endl;
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

void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &fx, std::vector<double> &fy, std::vector<double> &fz) {
    std::string junk;
    // Skip ID.
    traj_filestream >> junk;
    traj_filestream >> type[i];
    // Read (scaled) coordinate.
    traj_filestream >> x[i] >> y[i] >> z[i];
    // Skip force.
    traj_filestream >> fx[i]  >> fy[i]  >> fz[i];
}

void wrap_coordinate(double &coordinate, const double box_length) {
    while (coordinate < 0.0) {
        coordinate += box_length;
    }
    while (coordinate >= box_length) {
        coordinate -= box_length;
    }
}

void wrap_coordinate_slab(double &coordinate, const double slab_length) {
    while (coordinate < 0.0) {
        coordinate += slab_length;
    }
    while (coordinate >= slab_length) {
        coordinate -= slab_length;
    }
}

void min_image(double &displacement, const double box_length) {
    displacement -= box_length * round(displacement / box_length);
}
void min_image_slab(double &displacement, const double slab_length) {
    displacement -= slab_length * round(displacement / slab_length);
}
