__constant int Q = 9;
__constant int cx[9] = {0,1,0,-1,0,1,-1,-1,1};
__constant int cy[9] = {0,0,1,0,-1,1,1,-1,-1};
__constant float w[9] = {4.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,
                         1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f};

__kernel void boundaryConditions(
    __global float* f_new,
    __global float* ux,
    __global float* uy,
    __global uchar* flags,
    const float u_in_x,
    const float u_in_y,
    const float rho_out,
    const int NX,
    const int NY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= NX || y >= NY) return;
    
    int node = y * NX + x;
    
    if (flags[node] == 2) { // Inlet - OSTAJE ISTO
        float rho_in = 1.0f;
        
        float ux_in, uy_in;
        
        if (x == 0) { // Levi ulaz
            float height_factor = 1.0f - 4.0f * (y - NY/2.0f) * (y - NY/2.0f) / (NY * NY);
            height_factor = fmax(0.1f, height_factor); 
            ux_in = u_in_x * height_factor;
            uy_in = u_in_y * height_factor;
        }
        else if (x == NX-1) { // Desni ulaz  
            float height_factor = 1.0f - 4.0f * (y - NY/2.0f) * (y - NY/2.0f) / (NY * NY);
            height_factor = fmax(0.1f, height_factor);
            ux_in = -u_in_x * height_factor;
            uy_in = u_in_y * height_factor;
        }
        else if (y == 0) { // Donji ulaz
            float width_factor = 1.0f - 4.0f * (x - NX/2.0f) * (x - NX/2.0f) / (NX * NX);
            width_factor = fmax(0.1f, width_factor);
            ux_in = u_in_x * width_factor;
            uy_in = u_in_y * width_factor;
        }
        else { // Gornji ulaz
            float width_factor = 1.0f - 4.0f * (x - NX/2.0f) * (x - NX/2.0f) / (NX * NX);
            width_factor = fmax(0.1f, width_factor);
            ux_in = u_in_x * width_factor;
            uy_in = -u_in_y * width_factor;
        }
        
        ux[node] = ux_in;
        uy[node] = uy_in;
        
        float u2 = ux_in*ux_in + uy_in*uy_in;
        for (int i = 0; i < Q; i++) {
            float cu = cx[i]*ux_in + cy[i]*uy_in;
            f_new[i*NX*NY + node] = w[i]*rho_in*(1.f + 3.f*cu + 4.5f*cu*cu - 1.5f*u2);
        }
    }
    else if (flags[node] == 3) {
        
        float ux_out = 0.0f;
        float uy_out = 0.0f;
        
        int upstream = -1;
        if (x == NX-1 && x > 0) { // Desni outlet
            upstream = y*NX + (x-1);
            if (upstream >= 0 && upstream < NX*NY && flags[upstream] == 0) {
                ux_out = fmax(0.0f, ux[upstream]);
                uy_out = uy[upstream] * 0.9f; 
            }
        }
        else if (x == 0 && x < NX-1) { // Levi outlet  
            upstream = y*NX + (x+1);
            if (upstream >= 0 && upstream < NX*NY && flags[upstream] == 0) {
                ux_out = fmin(0.0f, ux[upstream]);
                uy_out = uy[upstream] * 0.9f;
            }
        }
        else if (y == NY-1 && y > 0) { // Gornji outlet
            upstream = (y-1)*NX + x;
            if (upstream >= 0 && upstream < NX*NY && flags[upstream] == 0) {
                ux_out = ux[upstream] * 0.9f;
                uy_out = fmax(0.0f, uy[upstream]);
            }
        }
        else if (y == 0 && y < NY-1) { // Donji outlet
            upstream = (y+1)*NX + x;
            if (upstream >= 0 && upstream < NX*NY && flags[upstream] == 0) {
                ux_out = ux[upstream] * 0.9f;
                uy_out = fmin(0.0f, uy[upstream]);
            }
        }
        
        ux_out = fmax(-0.3f, fmin(0.3f, ux_out));
        uy_out = fmax(-0.3f, fmin(0.3f, uy_out));
        
        ux[node] = ux_out;
        uy[node] = uy_out;
        
        float u2 = ux_out*ux_out + uy_out*uy_out;
        for (int i = 0; i < Q; i++) {
            float cu = cx[i]*ux_out + cy[i]*uy_out;
            f_new[i*NX*NY + node] = w[i]*rho_out*(1.f + 3.f*cu + 4.5f*cu*cu - 1.5f*u2);
        }
    }
}