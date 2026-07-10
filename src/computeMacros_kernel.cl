__constant int cx[9] = {0,1,0,-1,0,1,-1,-1,1};
__constant int cy[9] = {0,0,1,0,-1,1,1,-1,-1};

__kernel void computeMacros(
    __global const float* f,
    __global float* rho,
    __global float* ux,
    __global float* uy,
    __global const uchar* flags,
    const int NX,
    const int NY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= NX || y >= NY) return;
    
    int N = NX*NY;
    int node = y*NX + x;
    
    if(flags[node] == 1){
        rho[node] = 0.0f;
        ux[node] = 0.0f;
        uy[node] = 0.0f;
        return;
    }
    
    float rho_node = 0.0f;
    float ux_node = 0.0f;
    float uy_node = 0.0f;
    
    for(int i=0;i<9;i++){
        float fi = f[i*N+node];
        rho_node += fi;
        ux_node  += fi * cx[i];
        uy_node  += fi * cy[i];
    }
    
    if(rho_node > 1e-12f){
        ux_node /= rho_node;
        uy_node /= rho_node;
    } else {
        ux_node = 0.0f;
        uy_node = 0.0f;
    }
    
    rho[node] = rho_node;
    ux[node] = ux_node;
    uy[node] = uy_node;
}