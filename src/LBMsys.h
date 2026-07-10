#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

class LBMsys {
    int NX{};
    int NY{};
    int N {};

    std::vector<uint8_t> flags;
    std::vector<float> f;             // Q*N, SoA layout
    std::vector<float> f_new;         // buffer for streaming

    std::vector<float> rho_vec;
    std::vector<float> ux_vec;
    std::vector<float> uy_vec;

public:
    LBMsys() = default;

    int getSize() { return N; }
    int getWidth() { return NX; }
    int getHeight() { return NY; }

    std::vector<uint8_t>& getFlags() { return flags; }
    std::vector<float>& getF() { return f; }
    std::vector<float>& getFnew() { return f_new; }
    std::vector<float>& getRho() { return rho_vec; }
    std::vector<float>& getUx() { return ux_vec; }
    std::vector<float>& getUy() { return uy_vec; }

    inline int node_index(int x, int y) const {
        return y * NX + x;
    }

    inline float& f_at(int i, int node) {
        return f[i * N + node];
    }
    
    inline float& fnew_at(int i, int node) {
        return f_new[i * N + node];
    }

    inline uint8_t& flag_at(int node) {
        return flags[node];
    }

    float feq_at(int i, float rho, float ux, float uy) const;

    float rho_at(int node) const { return rho_vec[node]; }
    float ux_at(int node) const { return ux_vec[node]; }
    float uy_at(int node) const { return uy_vec[node]; }

    void loadConfig(std::string filename);
    void refineConfig(int k);
    void printFlags() const;
    void initialize(float u_in_x = 0.0f, float u_in_y = 0.0f); 

    void collide(float omega);  
    void stream();
    void applyBounceBack();
    void computeMacrosFrom(const std::vector<float>& src);
    void step(float omega, float u_in_x, float u_in_y, float rho_out = 1.0f);
    void boundaryConditions(float omegma, float u_in_x, float u_in_y, float rho_out = 1.0f);
};

struct LBMparams {
    static const int Q = 9;
    static constexpr int cx[Q] = { 0, 1, 0, -1,  0, 1, -1, -1,  1 };
    static constexpr int cy[Q] = { 0, 0, 1,  0, -1, 1,  1, -1, -1 };
    static constexpr float w[Q] = {
        4.0f / 9.0f,
        1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
    };
    static constexpr int opp[Q] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
};