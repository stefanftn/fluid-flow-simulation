__constant int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__constant int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

__kernel void bounceBack(
    __global const float* f,     
    __global float* f_new,       
    __global const uchar* flags,
    const int NX,
    const int NY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= NX || y >= NY) return;
    
    int N = NX * NY;
    int node = y * NX + x;
    
    if(flags[node] != 0) return;
    
    for(int i = 1; i < 9; i++) {
        int xn = x + cx[i];
        int yn = y + cy[i];
        
        if(xn >= 0 && xn < NX && yn >= 0 && yn < NY) {
            int neighbor = yn * NX + xn;
            
            if(flags[neighbor] == 1) {
                f_new[opp[i] * N + node] = f[i * N + node];
            }
        }
        else {
            f_new[opp[i] * N + node] = f[i * N + node];
        }
    }
}