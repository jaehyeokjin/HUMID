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
#include <numeric>
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
    int evb_number = -1;
    size_t n_hyd = 8;
    std::vector<size_t> evb_site(n_hyd, 0);
    std::string evb_name = argv[3];
    double evb_distance = stof(evb_name);
    double box_length = 19.73;
    double slab_length = 19.73;

    // Histogram specs

    std::vector<double> probability_22(n_sites, 0.0);
    std::ifstream traj_filestream;
    traj_filestream.open(traj_filename.c_str());

    std::string junk;
    size_t curr_timestep = 0;
    std::vector<size_t> idx(n_sites);
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
    std::vector<double> hyd_prob(n_sites);
    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());

    // For each frame in the trajectory
    size_t iframe = 0;
    while ((traj_filestream.good())) {
        // Read header lines.
        // Read timestep.
        curr_timestep = read_timestep(traj_filestream);
        // Skip the number of atoms.
        skip_natoms(traj_filestream);
        // Skip reading box dimensions.
        skip_box(traj_filestream);
        //std::cout << curr_timestep << std::endl;
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
            wrap_coordinate(z[i], box_length);
        }

        skip_line(traj_filestream);

        // Do distance analysis.
        // Populate the radial histogram.
        std::vector<int> neighbor_id(cg_site, 0);
        std::vector<double> neighbor_prob(cg_site, 0.0);
        for (size_t i = 0; i < cg_site; ++i) {
            double local_density_22 = 0.0;
            for (size_t j = 0; j < cg_site; ++j) {
                if (i != j)  {
                    double dx22 = x[i] - x[j];
                    double dy22 = y[i] - y[j];
                    double dz22 = z[i] - z[j];
                    min_image(dx22, box_length);
                    min_image(dy22, box_length);
                    min_image(dz22, box_length);
                    double distance_squared_22 = dx22 * dx22 + dy22 * dy22 + dz22 * dz22;
                    double distance = sqrt(distance_squared_22);
                    local_density_22 += 1.0/(1.0+exp(11.0*(distance-2.5)));
                }
            }
            probability_22[i] = 0.5*(1.0+tanh((local_density_22-1.25)/(0.18)));
            hyd_prob[i] = 0.5*(1.0+tanh((local_density_22-1.25)/(0.18)));
        }

        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
             [&probability_22](size_t i1, size_t i2) {return probability_22[i1] < probability_22[i2];});
        std::stable_sort(probability_22.begin(), probability_22.end());
        for (size_t k = 0; k < n_hyd; ++k) {
            evb_site[k] = idx[idx.size()-1-k];
            type[evb_site[k]] = 2;
        }
        if (evb_number > -1){
            for (size_t k = 0; k < n_hyd; ++k){
                for (size_t i = 0; i < cg_site; ++i) {
                    double dx22 = x[i] - x[evb_site[k]];
                    double dy22 = y[i] - y[evb_site[k]];
                    double dz22 = z[i] - z[evb_site[k]];
                    min_image(dx22, box_length);
                    min_image(dy22, box_length);
                    min_image(dz22, box_length);
                    double distance_squared_22 = dx22 * dx22 + dy22 * dy22 + dz22 * dz22;
                    double distance = sqrt(distance_squared_22);
                    if (distance < evb_distance){
                        neighbor_id[i]=i;
                        neighbor_prob[i]=hyd_prob[i];
                    }
                    else{
                        neighbor_id[i]=-1;
                        neighbor_prob[i]=-1.0;
                    }
                }
                size_t max_id = 0;
                double max_val = 0.0;

                for (size_t i = 0; i<cg_site; ++i){
                    if (neighbor_prob[i] > max_val){
                        max_val = neighbor_prob[i];
                        max_id  = neighbor_id[i];
                    }
                }
                type[max_id] = 2;
                evb_site[k] = max_id;
            }
        }
        evb_number = 1;
	output_filestream << "ITEM: TIMESTEP\n" << curr_timestep << "\nITEM: NUMBER OF ATOMS\n1000\nITEM: BOX BOUNDS pp pp pp\n0.00000 19.73\n0.00000 19.73\n0.00000 19.73\nITEM: ATOMS id type x y z fx fy fz"  << std::endl;
        for (size_t j = 0; j < cg_site; ++j){
            size_t index_22 = j+1;
            output_filestream << index_22 << " " << type[j] << " " << std::setprecision(6) << std::fixed << xu[j] << " " << yu[j] << " " << zu[j] << " " << std::setprecision(10) << std::fixed << fx[j]<< " " << fy[j]<< " " << fz[j] << std::endl;
        }
        iframe++;
        if (iframe%1000 == 0){
            std::cout << iframe << " is processed!" << std::endl;
            //std::cout << evb_number << "is evb number and evb distance is " << evb_distance << std::endl;
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
