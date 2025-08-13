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
void read_body_line_aa(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w);

void wrap_coordinate(double &coordinate, const double box_length);
void min_image(double &displacement, const double box_length);


int main(int argc, char *argv[]) {
    std::cout << "Hello!" << std::endl;

    std::string traj_filename = argv[1];
    std::string output_filename = argv[2];


    // System specs
    size_t n_sites = 256;
    size_t cg_site = 256;
    double box_length = 19.7320;

    // Histogram specs
    size_t n_histogram_bins = 601;
    double rdf_cut = 4.50;
    double histogram_binwidth = 0.05;
    std::vector<double> histogram_radii(n_histogram_bins);
    std::vector<double> probability_donor(n_histogram_bins, 0.0);
    std::vector<double> probability_acceptor(n_histogram_bins, 0.0);
    std::vector<double> counting_vector(n_histogram_bins, 0);

    for (size_t i = 0; i < n_histogram_bins; ++i) {
        histogram_radii[i] = ((double) i) * histogram_binwidth;
        // std::cout << "Bin " << i << " has radius " << histogram_radii[i] << " and shell volume " << histogram_vols[i] << std::endl;
    }

    std::ifstream traj_filestream;
    traj_filestream.open(traj_filename.c_str());

    std::string junk;
    size_t curr_timestep = 0;

    std::vector<size_t> type(n_sites);
    std::vector<double> x(n_sites);
    std::vector<double> y(n_sites);
    std::vector<double> z(n_sites);
    std::vector<double> w(n_sites);


    // For each frame in the trajectory
    size_t iframe = 0;
    while ((traj_filestream.good()) && (iframe < 9999)) {
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
            read_body_line(traj_filestream, i, type, x, y, z, w);
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
        for (size_t i = 0; i < cg_site; ++i) {
            double n_coordination = 0.0;
            double n_acceptor = 0.0;
            double n_donor = 0.0;
            for (size_t j = 0; j < cg_site; ++j) {
                if (i != j)  {
                    // Calculate the distance squared.
                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];
                    double dz = z[i] - z[j];
                    min_image(dx, box_length);
                    min_image(dy, box_length);
                    min_image(dz, box_length);
                    double distance_squared = dx * dx + dy * dy + dz * dz;
                    // Accumulate in histogram if it belongs.
                    if (distance_squared < rdf_cut * rdf_cut * 2.0) {
                        double distance = sqrt(distance_squared);
                        double ptcl_contribution = 0.5 * (1.0 - tanh((distance - rdf_cut) / (0.1 * distance)));
                        n_coordination += ptcl_contribution;
                    }
                }
            }
            size_t hist_bin = (size_t) floor((n_coordination) / histogram_binwidth);
            counting_vector[hist_bin] += 1;
        }
        iframe++;
        std::cout << iframe << " is processed!" << std::endl;
    }
    // Averaging the final value
    // Print out the final g(r)
    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());
    for (size_t i = 0; i < n_histogram_bins; ++i) {
        output_filestream << histogram_radii[i] << " " << counting_vector[i] << std::endl;
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
    traj_filestream >> junk  >> junk  >> junk ;
}

void wrap_coordinate(double &coordinate, const double box_length) {
    while (coordinate < 0.0) {
        coordinate += box_length;
    }
    while (coordinate >= box_length) {
        coordinate -= box_length;
    }
}

void min_image(double &displacement, const double box_length) {
    displacement -= box_length * round(displacement / box_length);
}
