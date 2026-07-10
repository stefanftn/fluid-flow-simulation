#include "LBMsys.h"
#include <fstream>
#include <sstream>
#include <iostream>

void LBMsys::loadConfig(std::string filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(ifs, line)) {
        line_num++;
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::vector<int> row;
        int val;
        while (iss >> val) {
            if (val < 0 || val > 3) {
                throw std::runtime_error("Invalid flag at line " + std::to_string(line_num));
            }
            row.push_back(val);
        }

        if (row.empty()) continue;

        if (NX == 0) {
            NX = (int)row.size();
        }
        else if ((int)row.size() != NX) {
            throw std::runtime_error("Non-rectangular config at line " + std::to_string(line_num));
        }

        for (int v : row) {
            flags.push_back((uint8_t)v);
        }
        NY++;
    }

    if (NX <= 0 || NY <= 0) {
        throw std::runtime_error("Empty or invalid config file");
    }

    if ((int)flags.size() != NX * NY) {
        throw std::runtime_error("Internal size mismatch after parsing config");
    }

    N = NX * NY;
}

void LBMsys::refineConfig(int k) {
    if (k <= 1) return;

    int newNX = (NX - 2) * k + 2;
    int newNY = (NY - 2) * k + 2;
    std::vector<uint8_t> refined(newNX * newNY, 0);

    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            uint8_t v = flags[y * NX + x];
            int startX = (x == 0) ? 0 : (x == NX - 1 ? newNX - 1 : (x - 1) * k + 1);
            int startY = (y == 0) ? 0 : (y == NY - 1 ? newNY - 1 : (y - 1) * k + 1);
            int blockW = (x == 0 || x == NX - 1) ? 1 : k;
            int blockH = (y == 0 || y == NY - 1) ? 1 : k;

            for (int dy = 0; dy < blockH; dy++) {
                for (int dx = 0; dx < blockW; dx++) {
                    refined[(startY + dy) * newNX + (startX + dx)] = v;
                }
            }
        }
    }

    NX = newNX;
    NY = newNY;
    N = NX * NY;
    flags = refined;
    f.assign(LBMparams::Q * N, 0.0f);
    f_new.assign(LBMparams::Q * N, 0.0f);
}

float LBMsys::feq_at(int i, float rho, float ux, float uy) const {
    float cu = LBMparams::cx[i] * ux + LBMparams::cy[i] * uy;
    float u2 = ux * ux + uy * uy;
    return LBMparams::w[i] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u2);
}

void LBMsys::initialize(float u_in_x, float u_in_y) {
    rho_vec.assign(N, 1.0f);
    ux_vec.assign(N, 0.0f);
    uy_vec.assign(N, 0.0f);

    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            int node = node_index(x, y);

            if (flags[node] == 2) { // inlet
                // Prilagodi brzinu u zavisnosti od pozicije
                if (x == 0) { // Levi ulaz
                    ux_vec[node] = u_in_x;
                    uy_vec[node] = u_in_y;
                }
                else if (x == NX - 1) { // Desni ulaz
                    ux_vec[node] = -u_in_x;
                    uy_vec[node] = u_in_y;
                }
                else if (y == 0) { // Donji ulaz
                    ux_vec[node] = u_in_x;
                    uy_vec[node] = u_in_y;
                }
                else if (y == NY - 1) { // Gornji ulaz
                    ux_vec[node] = u_in_x;
                    uy_vec[node] = -u_in_y;
                }
            }
            else {
                ux_vec[node] = 0.0f;
                uy_vec[node] = 0.0f;
            }
            rho_vec[node] = 1.0f;

            for (int i = 0; i < LBMparams::Q; ++i) {
                f[i * N + node] = feq_at(i, rho_vec[node], ux_vec[node], uy_vec[node]);
            }
        }
    }
    f_new = f;
}
void LBMsys::collide(float omega) {
    for (int node = 0; node < N; ++node) {
        if (flags[node] == 1) { // solid node
            for (int i = 0; i < LBMparams::Q; ++i) {
                f[i * N + node] = f_new[i * N + node]; // keep bounced-back values
            }
            continue;
        }

        float rho = rho_vec[node];
        float ux = ux_vec[node];
        float uy = uy_vec[node];

        // relaxation
        for (int i = 0; i < LBMparams::Q; ++i) {
            float feq = feq_at(i, rho, ux, uy);
            float fin = f_new[i * N + node]; // post-stream, pre-collision
            f[i * N + node] = (1.0f - omega) * fin + omega * feq;
        }
    }
}

void LBMsys::stream() {
    std::fill(f_new.begin(), f_new.end(), 0.0f);

    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            int node = node_index(x, y);

            for (int i = 0; i < LBMparams::Q; i++) {
                int xn = x + LBMparams::cx[i];
                int yn = y + LBMparams::cy[i];

                if (xn >= 0 && xn < NX && yn >= 0 && yn < NY) {
                    int nnode = node_index(xn, yn);
                    f_new[i * N + nnode] = f[i * N + node];
                }
            }
        }
    }
}

void LBMsys::applyBounceBack() {
    for (int node = 0; node < N; node++) {
        if (flags[node] == 1) { // solid node
            for (int i = 0; i < LBMparams::Q; i++) {
                f_new[i * N + node] = f[LBMparams::opp[i] * N + node];
            }
        }
    }
}

void LBMsys::boundaryConditions(float omega, float u_in_x, float u_in_y, float rho_out) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int node = node_index(x, y);

            if (flags[node] == 2) { // Inlet
                float rho_in = 1.0f;
                float ux_in = 0.0f;
                float uy_in = 0.0f;

                if (x == 0) { // Left inlet
                    float height_factor = 1.0f - 4.0f * (y - NY / 2.0f) * (y - NY / 2.0f) / (NY * NY);
                    height_factor = std::max(0.1f, height_factor);
                    ux_in = u_in_x * height_factor;
                    uy_in = u_in_y * height_factor;
                }
                else if (x == NX - 1) { // Right inlet
                    float height_factor = 1.0f - 4.0f * (y - NY / 2.0f) * (y - NY / 2.0f) / (NY * NY);
                    height_factor = std::max(0.1f, height_factor);
                    ux_in = -u_in_x * height_factor;
                    uy_in = u_in_y * height_factor;
                }
                else if (y == 0) { // Bottom inlet
                    float width_factor = 1.0f - 4.0f * (x - NX / 2.0f) * (x - NX / 2.0f) / (NX * NX);
                    width_factor = std::max(0.1f, width_factor);
                    ux_in = u_in_x * width_factor;
                    uy_in = u_in_y * width_factor;
                }
                else { // Top inlet
                    float width_factor = 1.0f - 4.0f * (x - NX / 2.0f) * (x - NX / 2.0f) / (NX * NX);
                    width_factor = std::max(0.1f, width_factor);
                    ux_in = u_in_x * width_factor;
                    uy_in = -u_in_y * width_factor;
                }

                ux_vec[node] = ux_in;
                uy_vec[node] = uy_in;

                float u2 = ux_in * ux_in + uy_in * uy_in;
                for (int i = 0; i < LBMparams::Q; ++i) {
                    float cu = LBMparams::cx[i] * ux_in + LBMparams::cy[i] * uy_in;
                    f_new[i * N + node] = LBMparams::w[i] * rho_in * (1.f + 3.f * cu + 4.5f * cu * cu - 1.5f * u2);
                }
            }
            else if (flags[node] == 3) { // Outlet
                float ux_out = 0.0f;
                float uy_out = 0.0f;

                int upstream = -1;
                if (x == NX - 1 && x > 0) upstream = node_index(x - 1, y);      // Right outlet
                else if (x == 0 && x < NX - 1) upstream = node_index(x + 1, y);  // Left outlet
                else if (y == NY - 1 && y > 0) upstream = node_index(x, y - 1); // Top outlet
                else if (y == 0 && y < NY - 1) upstream = node_index(x, y + 1); // Bottom outlet

                if (upstream != -1 && flags[upstream] == 0) { 
                    ux_out = ux_vec[upstream];
                    uy_out = uy_vec[upstream];

                    ux_out *= 0.9f;
                    uy_out *= 0.9f;

                    ux_out = std::max(-0.3f, std::min(0.3f, ux_out));
                    uy_out = std::max(-0.3f, std::min(0.3f, uy_out));
                }

                ux_vec[node] = ux_out;
                uy_vec[node] = uy_out;

                float u2 = ux_out * ux_out + uy_out * uy_out;
                for (int i = 0; i < LBMparams::Q; ++i) {
                    float cu = LBMparams::cx[i] * ux_out + LBMparams::cy[i] * uy_out;
                    f_new[i * N + node] = LBMparams::w[i] * rho_out * (1.f + 3.f * cu + 4.5f * cu * cu - 1.5f * u2);
                }
            }
        }
    }
}


void LBMsys::computeMacrosFrom(const std::vector<float>& src) {
    rho_vec.assign(N, 0.0f);
    ux_vec.assign(N, 0.0f);
    uy_vec.assign(N, 0.0f);

    for (int node = 0; node < N; ++node) {
        if (flags[node] == 1) { // solid node
            rho_vec[node] = 0.0f;
            ux_vec[node] = 0.0f;
            uy_vec[node] = 0.0f;
            continue;
        }

        float rho = 0.0f;
        float ux = 0.0f;
        float uy = 0.0f;

        for (int i = 0; i < LBMparams::Q; ++i) {
            float fi = src[i * N + node];
            rho += fi;
            ux += fi * LBMparams::cx[i];
            uy += fi * LBMparams::cy[i];
        }

        if (rho <= 1e-12f) {
            rho = 1.0f; // Prevent division by zero
            ux = 0.0f;
            uy = 0.0f;
        }
        else {
            ux /= rho;
            uy /= rho;
        }

        rho_vec[node] = rho;
        ux_vec[node] = ux;
        uy_vec[node] = uy;
    }
}

void LBMsys::step(float omega, float u_in_x, float u_in_y, float rho_out) {
    stream();
    boundaryConditions(omega, u_in_x, u_in_y, rho_out);
    applyBounceBack();
    computeMacrosFrom(f_new);
    collide(omega);
}

void LBMsys::printFlags() const {
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            std::cout << (int)flags[y * NX + x] << " ";
        }
        std::cout << std::endl;
    }
}