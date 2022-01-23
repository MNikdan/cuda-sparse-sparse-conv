using namespace std;
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <cstdio>
#include <cudnn.h>

#define CHUNK_SIZE 8196

__global__ void sparse_convolution_2d(int H, int W, int KH, int KW, int RH, int RW, int IC, int sX_nzn, int sW_nzn, float* R, float* sX_val, int* sX_x_idx, int* sX_y_idx, int* sX_z_idx, float* sW_val, int* sW_x_idx, int* sW_y_idx, int* sW_z_idx) {
    int step_size = gridDim.x * blockDim.x;
    int steps = (sX_nzn + step_size - 1) / step_size;
    int offset = blockIdx.x * blockDim.x + threadIdx.x; 
    int index = offset;

    for (int step = 0; step < steps; step++) {
        if (index >= sX_nzn)
            break;
        
        float val = sX_val[index];
        int z = sX_z_idx[index];
        
        int w_start = sW_z_idx[z];
        int w_end = sW_z_idx[z + 1];
        
        for (int i = w_start; i < w_end; i++) {
            int rx = sX_x_idx[index] - sW_x_idx[i];
            int ry = sX_y_idx[index] - sW_y_idx[i];
            if (rx >= 0 && rx < RW && ry >= 0 && ry < RH)
                atomicAdd(&R[ry * RW + rx], sW_val[i] * val);
        }
        
        index += step_size;
    }
    
}

void init_3d_matrix(float *m, int N1, int N2, int N3, float sparsity) {
    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                float p = float(rand()) / float((RAND_MAX));
                if (p <= sparsity)
                    m[(n1 * N2 + n2) * N3 + n3] = 0.;
                else
                    m[(n1 * N2 + n2) * N3 + n3] = (rand() % 100 + 1) / 100.;
            }
        }
    }
}

void init_2d_matrix(float *m, int H, int W, float sparsity) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            float p = float(rand()) / float((RAND_MAX));
            if (p <= sparsity)
                m[h * W + w] = 0.;
            else
                m[h * W + w] = (rand() % 100 + 1) / 100.;
        }
    }
}

void print_array(int* arr, int l) {
    std::cout << "[";
    for (int i = 0; i < l; i++){
        std::cout << arr[i] << ",";
    }
    std::cout << "]\n";
}

void print_array(float* arr, int l) {
    std::cout << "[";
    for (int i = 0; i < l; i++){
        std::cout << arr[i] << ",";
    }
    std::cout << "]\n";
}

void print_2d_array(float* arr, int H, int W) {
    std::cout << "[";
    for (int h = 0; h < H; h++){
        std::cout << "[";
        for (int w = 0; w < W; w++)
            std::cout << arr[h * W + w] << ",";
        std::cout << "],";
    }
    std::cout << "]\n";
}

void print_3d_array(float* arr, int N1, int N2, int N3) {
    std::cout << "[";
    for (int n1 = 0; n1 < N1; n1++){
        std::cout << "[";
        for (int n2 = 0; n2 < N2; n2++){
            std::cout << "[";
            for (int n3 = 0; n3 < N3; n3++)
                std::cout << arr[(n1 * N2 + n2) * N3 + n3] << ",";
            std::cout << "],";
        }
        std::cout << "],";
    }
    std::cout << "]\n";
}

void flush_2d_array(float* arr, int H, int W, string file_name) {
    remove(file_name.c_str());
    ofstream ff(file_name);
    
    ff << "[";
    for (int h = 0; h < H; h++){
        ff << "[";
        for (int w = 0; w < W; w++) {
            ff << arr[h * W + w];
            ff << ",";
        }
        ff << "],";
    }
    ff << "]\n";
}

void flush_3d_array(float* arr, int N1, int N2, int N3, string file_name) {
    remove(file_name.c_str());
    ofstream ff(file_name);
    
    ff << "[";
    for (int n1 = 0; n1 < N1; n1++){
        ff << "[";
        for (int n2 = 0; n2 < N2; n2++){
            ff << "[";
            for (int n3 = 0; n3 < N3; n3++){
                ff << arr[(n1 * N2 + n2) * N3 + n3];
                ff << ",";
            }
            ff << "],";
        }
        ff << "],";
    }
    ff << "]\n";
    
    ff.close();
}

std::tuple<int, float*, int*, int*, int*> to_sparse_rep(float* M, int X, int Y, int Z, bool z_starts) {
    int nzn = 0;
    for (int x = 0; x < X; x++) {
        for (int y = 0; y < Y; y++) {
            for (int z = 0; z < Z; z++) {
                if (M[(z * Y + y) * X + x] != 0.)
                    nzn++;
            }
        }
    }
    

    float* val = (float*) malloc(nzn * sizeof(float));
    int* x_idx = (int*) malloc(nzn * sizeof(int));
    int* y_idx = (int*) malloc(nzn * sizeof(int));
    
    int* z_idx;
    if (z_starts)
        z_idx = (int*) malloc((Z + 1) * sizeof(int));
    else
        z_idx = (int*) malloc(nzn * sizeof(int));

    int cnt = 0;
    for (int z = 0; z < Z; z++) {
        if (z_starts)
            z_idx[z] = cnt;
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {
                float v = M[(z * Y + y) * X + x];
                if (v != 0.) {
                    val[cnt] = v;
                    x_idx[cnt] = x;
                    y_idx[cnt] = y;
                    
                    if (!z_starts)
                        z_idx[cnt] = z;
                    cnt++;
                }
            }
        }
    }
    
    if (z_starts)
        z_idx[Z] = nzn;
    
    return std::make_tuple(nzn, val, x_idx, y_idx, z_idx);
}

int main() {
    srand(18);

    int IC = 1024;
    int N = 32;
    int KN = 3;
    int RN = N - KN + 1;

    float* X = (float*) malloc(N * N * IC * sizeof(float));
    float* W = (float*) malloc(KN * KN * IC * sizeof(float));
    float* R = (float*) malloc(RN * RN * sizeof(float));
    init_3d_matrix(X, IC, N, N, 0.7);
    init_3d_matrix(W, IC, KN, KN, 0.9);
    init_2d_matrix(R, RN, RN, 1.);
    
    float *sX_val, *sW_val;
    int *sX_x_idx, *sX_y_idx, *sX_z_idx, *sW_x_idx, *sW_y_idx, *sW_z_idx;
    int sX_nzn, sW_nzn;
    std::tie(sX_nzn, sX_val, sX_x_idx, sX_y_idx, sX_z_idx) = to_sparse_rep(X, N, N, IC, false);
    std::tie(sW_nzn, sW_val, sW_x_idx, sW_y_idx, sW_z_idx) = to_sparse_rep(W, KN, KN, IC, true);
    
//     std::cout << "X" << std::endl;
//     print_3d_array(X, IC, N, N);
//     std::cout << sX_nzn << std::endl;
//     print_array(sX_val, sX_nzn);
//     print_array(sX_x_idx, sX_nzn);
//     print_array(sX_y_idx, sX_nzn);
//     print_array(sX_z_idx, sX_nzn);
    
//     std::cout << "W" << std::endl;
//     print_3d_array(W, IC, KN, KN);
//     std::cout << sW_nzn << std::endl;
//     print_array(sW_val, sW_nzn);
//     print_array(sW_x_idx, sW_nzn);
//     print_array(sW_y_idx, sW_nzn);
//     print_array(sW_z_idx, IC + 1);
    
    flush_3d_array(X, IC, N, N, "X.arr");
    flush_3d_array(W, IC, KN, KN, "W.arr");
    

    float *d_sX_val, *d_sW_val, *d_R;
    int *d_sX_x_idx, *d_sX_y_idx, *d_sX_z_idx, *d_sW_x_idx, *d_sW_y_idx, *d_sW_z_idx;
    cudaMalloc(&d_sX_val, sX_nzn * sizeof(float));
    cudaMalloc(&d_sX_x_idx, sX_nzn * sizeof(int));
    cudaMalloc(&d_sX_y_idx, sX_nzn * sizeof(int));
    cudaMalloc(&d_sX_z_idx, sX_nzn * sizeof(int));
    cudaMalloc(&d_sW_val, sW_nzn * sizeof(float));
    cudaMalloc(&d_sW_x_idx, sW_nzn * sizeof(int));
    cudaMalloc(&d_sW_y_idx, sW_nzn * sizeof(int));
    cudaMalloc(&d_sW_z_idx, (IC + 1) * sizeof(int));
    cudaMalloc(&d_R, RN * RN * sizeof(float));
    
    cudaMemcpy(d_sX_val, sX_val, sX_nzn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sX_x_idx, sX_x_idx, sX_nzn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sX_y_idx, sX_y_idx, sX_nzn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sX_z_idx, sX_z_idx, sX_nzn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sW_val, sW_val, sW_nzn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sW_x_idx, sW_x_idx, sW_nzn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sW_y_idx, sW_y_idx, sW_nzn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sW_z_idx, sW_z_idx, (IC + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, RN * RN * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // int THREADS = 1024;
    // int BLOCKS = (N + THREADS - 1) / THREADS;
    
    int THREADS = 32;
    int BLOCKS = (CHUNK_SIZE + THREADS - 1) / THREADS;
    
    sparse_convolution_2d<<<THREADS, BLOCKS>>>(N, N, KN, KN, RN, RN, IC, sX_nzn, sW_nzn, d_R, d_sX_val, d_sX_x_idx, d_sX_y_idx, d_sX_z_idx, d_sW_val, d_sW_x_idx, d_sW_y_idx, d_sW_z_idx);
    
    cudaMemcpy(R, d_R, RN * RN * sizeof(float), cudaMemcpyDeviceToHost);
//     std::cout << "R" << std::endl;
//     print_2d_array(R, RN, RN);
    flush_2d_array(R, RN, RN, "R.arr");
    
    
    
    cudaFree(d_sX_val);
    cudaFree(d_sX_x_idx);
    cudaFree(d_sX_y_idx);
    cudaFree(d_sX_z_idx);
    cudaFree(d_sW_val);
    cudaFree(d_sW_x_idx);
    cudaFree(d_sW_y_idx);
    cudaFree(d_sW_z_idx);
    cudaFree(d_R);
    
    
    return 0;
}
