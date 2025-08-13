#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>


#define PI 3.1415926535

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
    size_t n_histogram_bins = 360;
    double histogram_binwidth = 0.5;
    double wcut = 0.85;
    std::vector<double> histogram_vals(n_histogram_bins);
    std::vector<double> histogram_radii(n_histogram_bins);
    for (size_t i = 0; i < n_histogram_bins; ++i) {
        histogram_radii[i] = ((double) i ) * histogram_binwidth;
    }
    size_t n_histo = 0;
    std::ifstream traj_filestream;
    traj_filestream.open(traj_filename.c_str());
    std::string junk;
    size_t curr_timestep = 0;

    std::vector<size_t> type(n_sites);
    std::vector<double> x(n_sites);
    std::vector<double> y(n_sites);
    std::vector<double> z(n_sites);
    std::vector<double> w(n_sites);
    std::vector<double> pivot(3, 0.0);
    std::vector<std::vector <double> > delta_r(n_sites, std::vector<double>(3, 0.0));
    std::vector<size_t> sort_dist(n_sites);
    std::vector<double> dist_vals(n_sites);
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
                    if (distance_squared < 10.0*10.0) {
                        double distance = sqrt(distance_squared);
                        w_value += 1.0/(1.0+exp(11.0*(distance-2.5)));
                    }
                }
            }
            w[i] = w_value;
        }

        std::vector<double> dense_state_probability(n_sites, 1.0);
        // Scale all coordinates by the box size and wrap.
        for (size_t i = 0; i < n_sites; ++i) {
            std::vector<std::vector <double> > delta_r(n_sites, std::vector<double>(3, 0.0));
            std::iota(sort_dist.begin(),sort_dist.end(),0); 
            dense_state_probability[i] = 0.5 * (1.0 + tanh((w[i] - wcut) / (0.15 * wcut)));
            for (size_t j = 0; j < n_sites; ++j){
                if (i != j){
                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];
                    double dz = z[i] - z[j];
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
                    distance = sqrt(distance_squared);
                    delta_r[j][0]=dx/distance;
                    delta_r[j][1]=dy/distance;
                    delta_r[j][2]=dz/distance;
                    dist_vals[j] = distance;
    	        }
            else{
                    delta_r[j][0]=1;
                    delta_r[j][1]=1;
                    delta_r[j][2]=1;
                    double distance = box_length;
                    dist_vals[j] = distance;
                }
            }
            sort( sort_dist.begin(),sort_dist.end(), [&](int k,int j){return dist_vals[k]<dist_vals[j];} );
            double angle_1 = delta_r[sort_dist[0]][0]*delta_r[sort_dist[1]][0]+delta_r[sort_dist[0]][1]*delta_r[sort_dist[1]][1]+delta_r[sort_dist[0]][2]*delta_r[sort_dist[1]][2];
            double angle_2 = delta_r[sort_dist[0]][0]*delta_r[sort_dist[2]][0]+delta_r[sort_dist[0]][1]*delta_r[sort_dist[2]][1]+delta_r[sort_dist[0]][2]*delta_r[sort_dist[2]][2];
            double angle_3 = delta_r[sort_dist[2]][0]*delta_r[sort_dist[1]][0]+delta_r[sort_dist[2]][1]*delta_r[sort_dist[1]][1]+delta_r[sort_dist[2]][2]*delta_r[sort_dist[1]][2];
            double theta_1 = acos(angle_1)* 180.0 / PI;
            double theta_2 = acos(angle_2)* 180.0 / PI;
            double theta_3 = acos(angle_3)* 180.0 / PI;
            size_t hist_bin1 = (size_t) floor(theta_1 / histogram_binwidth);
            size_t hist_bin2 = (size_t) floor(theta_2 / histogram_binwidth);
            size_t hist_bin3 = (size_t) floor(theta_3 / histogram_binwidth);
            histogram_vals[hist_bin1]+= dense_state_probability[i];
            histogram_vals[hist_bin2]+= dense_state_probability[i];
            histogram_vals[hist_bin3]+= dense_state_probability[i];
            n_histo += 3.0*dense_state_probability[i];
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
    for (size_t i = 0; i < n_histogram_bins; ++i) {
        output_filestream << histogram_radii[i] << " " << histogram_vals[i]/((double) n_histo) << std::endl;
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

