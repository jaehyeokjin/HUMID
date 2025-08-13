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
void read_body_line(std::ifstream &traj_filestream, const size_t i, std::vector<size_t> &type, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w);

void wrap_coordinate(double &coordinate, const double box_length);
void min_image(double &displacement, const double box_length);

double get_weight(const size_t particle_id, const Species species, const std::vector<double> &dense_state_probabilities);

int main(int argc, char *argv[]) {
    std::cout << "Hello!" << std::endl;

    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << ": trajectory_filename output_filename pivot_species other_species" << std::endl;
        exit(EXIT_SUCCESS);
    }

    std::string traj_filename = argv[1];
    std::string output_filename = argv[2];

    Species pivot_species = parse_species(argv[3]);
    Species other_species = parse_species(argv[4]);

    // System specs
    size_t n_sites = 256;
    double box_length = 19.73;
    double box_volume = box_length * box_length * box_length;
    double wcut = 0.8;

    // Histogram specs
    size_t n_histogram_bins = 501;
    double histogram_cutoff = 10.0;
    double histogram_cutoff_squared = histogram_cutoff * histogram_cutoff;
    double histogram_binwidth = 0.02;
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
    while (traj_filestream.good()) {
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


        // Do g(r) calculation.

        // Calculate state probabilities.
        std::vector<double> dense_state_probability(n_sites, 1.0);
        for (size_t i = 0; i < n_sites; ++i) {
            dense_state_probability[i] = 0.5 * (1.0 + tanh((w[i] - wcut) / (0.1 * wcut)));
        }
        // Reset histogram increment to zero.
        for (size_t i = 0; i < n_histogram_bins; ++i) {
            histogram_increment_vals[i] = 0.0;
        }

        double pivot_weight = 0.0;
        double pivot_weight_as_other = 0.0;
        double pivot_sites = 0.0;
        double other_sites = 0.0;

        // Populate the radial histogram.
        for (size_t i = 0; i < n_sites; ++i) {

            if ((other_species != kN && type[i] == 1) || (other_species == kN && type[i] == 2)) {
                pivot_weight_as_other = get_weight(i, other_species, dense_state_probability); 
                other_sites += pivot_weight_as_other;
            }

            if ((pivot_species != kN && type[i] == 1) || (pivot_species == kN && type[i] == 2)) {
                
                pivot_weight = get_weight(i, pivot_species, dense_state_probability); 
                pivot_sites += pivot_weight;

                if (pivot_species == other_species) {
                    for (size_t j = i + 1; j < n_sites; ++j) {
                        if ((other_species != kN && type[j] == 1) || (other_species == kN && type[j] == 2))  {
                            // Calculate the distance squared.
                            double dx = x[i] - x[j];
                            double dy = y[i] - y[j];
                            double dz = z[i] - z[j];
                            min_image(dx, box_length);
                            min_image(dy, box_length);
                            min_image(dz, box_length);
                            double distance_squared = dx * dx + dy * dy + dz * dz;
                            // Accumulate in histogram if it belongs.
                            if (distance_squared < histogram_cutoff_squared) {
                                double distance = sqrt(distance_squared);
                                size_t hist_bin = (size_t) floor(distance / histogram_binwidth);
                                
                                double other_weight = get_weight(j, other_species, dense_state_probability); 
                                histogram_increment_vals[hist_bin] += 2.0 * pivot_weight * other_weight;
                            }
                        }
                    }
                } else {
                    for (size_t j = 0; j < n_sites; ++j) {
                        if (j != (i+1) && ((other_species != kN && type[j] == 1) || (other_species == kN && type[j] == 2)))  {
                            // Calculate the distance squared.
                            double dx = x[i] - x[j];
                            double dy = y[i] - y[j];
                            double dz = z[i] - z[j];
                            min_image(dx, box_length);
                            min_image(dy, box_length);
                            min_image(dz, box_length);
                            double distance_squared = dx * dx + dy * dy + dz * dz;
                            // Accumulate in histogram if it belongs.
                            if (distance_squared < histogram_cutoff_squared) {
                                double distance = sqrt(distance_squared);
                                size_t hist_bin = (size_t) floor(distance / histogram_binwidth);

                                double other_weight = get_weight(j, other_species, dense_state_probability); 
                                histogram_increment_vals[hist_bin] += pivot_weight * other_weight;
                            }
                        }
                    }  
                }
            }
        }

        // Normalize this frame's radial histogram.
        double other_number_density = 0.0;
        if (other_species == pivot_species) {
            other_number_density = (pivot_sites - 1.0) / box_volume;
        } else {
            other_number_density = (other_sites) / box_volume;            
        }
        double curr_pivot_number = pivot_sites;
        for (size_t i = 0; i < n_histogram_bins; ++i) {
            histogram_increment_vals[i] /= histogram_vols[i] * other_number_density * curr_pivot_number; 
        }

        // Accumulate with previous framewise rdfs.
        for (size_t i = 0; i < n_histogram_bins; ++i) {
            histogram_vals[i] = (last_total_pivot_number * histogram_vals[i] + curr_pivot_number * histogram_increment_vals[i]) / (last_total_pivot_number + curr_pivot_number);
        }
        last_total_pivot_number += curr_pivot_number;
        iframe++;
        std::cout << iframe << " is processed!" << std::endl;
    }

    // Print out the final g(r)
    std::ofstream output_filestream;
    output_filestream.open(output_filename.c_str());
    for (size_t i = 0; i < n_histogram_bins; ++i) {
        output_filestream << histogram_radii[i] << " " << histogram_vals[i] << std::endl;
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
