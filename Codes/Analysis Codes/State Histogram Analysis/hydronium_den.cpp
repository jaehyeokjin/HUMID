#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

typedef enum species {kL, kD, kM, kN} Species;

Species parse_species(const std::string species_identifier) {
    if (species_identifier == "L") {
        return kL;
    } else if (species_identifier == "D") {
        return kD;
    } else if (species_identifier == "M") {
        return kM;
    } else if (species_identifier == "N") {
        return kN;
    } else {
        std::cout << "Each Species must be one of L, D, M, or N." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void skip_line(std::ifstream &traj_filestream);
size_t read_timestep(std::ifstream &traj_filestream);
void skip_natoms(std::ifstream &traj_filestream);
void skip_box(std::ifstream &traj_filestream);
void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z);

void wrap_coordinate(double &coordinate, const double box_length);
void min_image(double &displacement, const double box_length);

double get_weight(const size_t particle_id, const Species species, const std::vector<double> &dense_state_probabilities);

int main(int argc, char *argv[]) {
    std::cout << "Hello!" << std::endl;

    std::string traj_filename = argv[1];
    std::string output_filename = argv[2];
    double cut_off = atof(argv[3]);
    double wcut = atof(argv[4]);
    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());

    // System specs
    size_t n_sites = 256;
    double box_length = 19.73;
    double box_volume = box_length * box_length * box_length;


    // Histogram specs
    size_t n_histogram_bins = 101;
    double histogram_cutoff = 10.0;
    double histogram_cutoff_squared = histogram_cutoff * histogram_cutoff;
    double histogram_binwidth = 0.1;
    std::vector<double> histogram_radii(n_histogram_bins);
    std::vector<double> histogram_vals(n_histogram_bins, 0.0);
    std::vector<double> histogram_increment_vals(n_histogram_bins, 0.0);
    std::vector<double> histogram_vols(n_histogram_bins);

    for (size_t i = 0; i < n_histogram_bins; ++i) {
        histogram_radii[i] = ((double) i + 0.5) * histogram_binwidth;
        histogram_vols[i] = (4.0 / 3.0) * M_PI * pow(histogram_binwidth, 3.0) * ((double) (3 * i * (i + 1) + 1));
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
    double last_total_pivot_number = 0.0;
    while ((traj_filestream.good()) && (iframe <10000)) {
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
            read_body_line(traj_filestream, i, type, x, y, z);
        }

        // Scale all coordinates by the box size and wrap.
        for (size_t i = 0; i < n_sites; ++i) {
            wrap_coordinate(x[i], box_length);
            wrap_coordinate(y[i], box_length);
            wrap_coordinate(z[i], box_length);
        }
        skip_line(traj_filestream);


        // Do g(r) calculation.

        for (size_t i = 0; i < n_sites; ++i){
            double w_value = 0.0;
            for (size_t j = 0; j<n_sites;++j){
                if (i!=j){
                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];
                    double dz = z[i] - z[j];
                    min_image(dx, box_length);
                    min_image(dy, box_length);
                    min_image(dz, box_length);
                    double distance_squared = dx * dx + dy * dy + dz * dz;
                    if (distance_squared < histogram_cutoff_squared) {
                        double distance = sqrt(distance_squared);
                        w_value += 1.0/(1.0+exp(11.0*(distance-2.5)));
                    }                
                }
            }
            w[i] = w_value;
        }

        // Calculate state probabilities.
        std::vector<double> dense_state_probability(n_sites, 1.0);
        for (size_t i = 0; i < n_sites; ++i) {
            dense_state_probability[i] = 0.5 * (1.0 + tanh((w[i] - wcut) / (cut_off * wcut)));
        }
        // Reset histogram increment to zero.
        for (size_t i = 0; i < n_histogram_bins; ++i) {
            histogram_increment_vals[i] = 0.0;
        }
        double max_wi = 0.0;
        double sum_wi = 0.0;
        // Populate the radial histogram.
        for (size_t i = 0; i < n_sites; ++i) {
            sum_wi += dense_state_probability[i];
            if (dense_state_probability[i] > max_wi) {
                max_wi = dense_state_probability[i];
            }
        }
        iframe++;
        output_filestream << iframe << " " << sum_wi << " " << max_wi <<  std::endl;
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

void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z) {
    std::string junk;
    // Skip ID.
    traj_filestream >> junk;
    // Read type.
    traj_filestream >> type[i];
    // Read (scaled) coordinate.
    traj_filestream >> x[i] >> y[i] >> z[i];
    // Skip velocity.
    traj_filestream >> junk  >> junk  >> junk;
    // Read neopentane density, skip methanol density.
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

double get_weight(const size_t particle_id, const Species species, const std::vector<double> &dense_state_probability) {
    if (species == kD) {
        return dense_state_probability[particle_id];
    } else if (species == kL) {
        return 1.0 - dense_state_probability[particle_id];
    } else {
        return 1.0;
    }
}
