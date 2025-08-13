#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <string>

struct Particle {
    double x, y, z;
};

double squaredDistance(const Particle& a, const Particle& b, const std::vector<double>& box) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    dx -= round(dx / box[0]) * box[0];
    dy -= round(dy / box[1]) * box[1];
    dz -= round(dz / box[2]) * box[2];
    return dx * dx + dy * dy + dz * dz;
}

int main() {
    std::string trajectoryFile = "../250.lammpstrj";
    std::string outputFile = "msd_output.txt";
    std::ifstream inFile(trajectoryFile);
    std::ofstream outFile(outputFile);
    std::string line;
    std::vector<double> boxSize(3);
    std::vector<Particle> type2Positions; // Only store type 2 particle positions
    int snapshotCount = 0;
    int currentTimestep;

    while (std::getline(inFile, line)) {
        if (line.find("ITEM: TIMESTEP") != std::string::npos) {
            std::getline(inFile, line);
            currentTimestep = std::stoi(line);
            std::cout << "Processing timestep: " << currentTimestep << std::endl;
            snapshotCount++;
            if (snapshotCount > 1000000) break;
            continue;
        }
        if (line.find("ITEM: BOX BOUNDS") != std::string::npos) {
            for (int i = 0; i < 3; ++i) {
                std::getline(inFile, line);
                std::istringstream iss(line);
                double min, max;
                iss >> min >> max;
                boxSize[i] = max - min;
            }
            continue;
        }
        if (line.find("ITEM: ATOMS") != std::string::npos) {
            while (std::getline(inFile, line) && !line.empty()) {
                std::istringstream iss(line);
                int id, type;
                Particle p;
                iss >> id >> type >> p.x >> p.y >> p.z;
                if (type == 2) {
                    type2Positions.push_back(p); // Store position of type 2 particle
                    break; // Stop reading further as we found our type 2 particle
                }
            }
        }
    }
    for (int dt = 1; dt <= 10000; ++dt) {
        double totalMSD = 0.0;
        int validCounts = 0;
        for (int t = 0; t + dt < type2Positions.size(); ++t) {
            totalMSD += squaredDistance(type2Positions[t], type2Positions[t + dt], boxSize);
            validCounts++;
        }
        if (validCounts > 0) {
            double avgMSD = totalMSD / validCounts;
            outFile << dt << " " << avgMSD << "\n";
        }
    }

    inFile.close();
    outFile.close();
    return 0;
}

