__constant float cx[9] = {0,1,0,-1,0,1,-1,-1,1};
__constant float cy[9] = {0,0,1,0,-1,1,1,-1,-1};
__constant float w[9] = {4.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/36,1.0f/36,1.0f/36,1.0f/36};

__kernel void collide(
    __global float* f,                   // izlaz
    __global const float* f_new,         // ulaz
    __global float* rho,
    __global float* ux,
    __global float* uy,
    __global const uchar* flags,
    const float omega,
    const int NX,
    const int NY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= NX || y >= NY) return;

    int N = NX * NY;
    int node = y * NX + x;

    if (flags[node] == 1) { // solid
        for (int i = 0; i < 9; i++) {
            f[i*N + node] = f_new[i*N + node];
        }
        return;
    }

    float rho_node = rho[node];
    float ux_node  = ux[node];
    float uy_node  = uy[node];

    for (int i = 0; i < 9; i++) {
        float cu = cx[i]*ux_node + cy[i]*uy_node;
        float u2 = ux_node*ux_node + uy_node*uy_node;
        float feq = w[i]*rho_node*(1.0f + 3.0*cu + 4.5f*cu*cu - 1.5f*u2);
        f[i*N + node] = f_new[i*N + node]*(1.0f - omega) + omega*feq;
    }
}
