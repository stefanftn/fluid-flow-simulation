__constant int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

__kernel void stream(
    __global const float* f,   // Input
    __global float* f_new,             // Output
    __global const uchar* flags,
    const int NX,
    const int NY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= NX || y >= NY) return;
    
    int N = NX*NY;
    int node = y*NX + x;
    
    for(int i=0; i<9; i++) {
        int xn = x + cx[i];
        int yn = y + cy[i];
        
        if(xn >= 0 && xn < NX && yn >= 0 && yn < NY) {
            int nnode = yn*NX + xn;
            if(flags[nnode] != 1) {  // Skip solid nodes
                f_new[i*N+nnode] = f[i*N+node];
            }
        }
    }
}