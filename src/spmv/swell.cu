#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "spmv.h"
#include "spmv_util.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"





// CSR SpMV kernels based on a scalar model (one thread per row)
// Straightforward translation of standard CSR SpMV to CUDA
// where each thread computes y[i] += A[i,:] * x 
// (the dot product of the i-th row of A with the x vector)

// anything accessed using offset needs to be transformed.

__global__ void spmv_csr_scalar(int N, const int* Ap, 
                                const int* Aj, const ValueT * Ax, 
                                const ValueT * x, ValueT * y, int* tmap, int* warp_base, int* nnz) {

     __shared__ int sh_offsets[32];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N) return;

    int nid = tmap[id];

    int wid = id / 32;
    int lwid = threadIdx.x / 32;

    int wsz = min(32, N - wid*32);


    if(id%32 == 0){
        sh_offsets[lwid] = 0;
    }
    
    int wbase = warp_base[wid] + id%32;
    int lim = nnz[id];

    if(0 == nnz[id])
        atomicAdd(&sh_offsets[lwid], 1);
    //  __syncwarp();

    int row_begin = Ap[nid];
    int row_end = Ap[nid+1];
    ValueT sum = 0;
    for(int i = 0; i < lim;){
        sum += Ax[wbase] * x[Aj[wbase]];

        wbase -= sh_offsets[lwid];
        i++;
        if(i == nnz[id])
            atomicAdd(&sh_offsets[lwid], 1);
        // __syncwarp();
        wbase += wsz;
    }
    y[nid] = sum;
}


__global__ void populate(int N, int* row_offsets, ValueT* Ax, int* Aj, ValueT* nAx, int* nAj, int* tmap, int* nnz, int* warp_base, int* d_check){
    __shared__ int sh_offsets[32];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N) return;
    // check if its a full warp

    int nid = tmap[id];

    int wid = id / 32;
    int lwid = threadIdx.x / 32;

    int wsz = min(32, N - wid*32);


    if(id%32 == 0){
        sh_offsets[lwid] = 0;
    }
    
    int wbase = warp_base[wid] + id%32;
    atomicAdd(&sh_offsets[lwid], (int)(0 == nnz[id]));
     __syncwarp();
    // sh_offsets[wid] += 0 == nnz[id];
    // for (int iw = 16; iw > 0; iw = iw / 2)
    // {
    //     sh_offset = max(sh_offset, __shfl_down_sync(-1, sh_offset, iw));
    // }
    // __syncwarp();

    for(int i = 0;i < nnz[id];){
        nAx[wbase] = Ax[i + row_offsets[nid]];
        nAj[wbase] = Aj[i + row_offsets[nid]];

        atomicAdd(&d_check[wbase],1);
        wbase -= sh_offsets[lwid];
        // warp shuffle, find maximum sh_offset in the warp
        i++;
        atomicAdd(&sh_offsets[lwid], (int)(i == nnz[id]));
        // sh_offsets[wid] += i == nnz[id];
        __syncwarp();
        wbase += wsz;
    }
}

void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) {
  auto N = g.V();
  auto m  = N;
  auto nnz_tot = g.E();
	auto h_Ap = g.in_rowptr();
	auto h_Aj = g.in_colidx();	
	//print_device_info(0);
	int *d_Ap;
    int *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz_tot * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz_tot * sizeof(int), cudaMemcpyHostToDevice));

	ValueT *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz_tot));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz_tot * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
	for(int i = 0; i < m; i ++) y_copy[i] = h_y[i];
	// SpmvSerial(m, nnz_tot, h_Ap, h_Aj, h_Ax, h_x, y_copy);
    

    int* nnz;
    int* tmap;

    nnz = (int*)malloc(N * sizeof(int));
    
    tmap = (int*)malloc(N * sizeof(int));

    for(int i=0;i<N;i++){
        nnz[i] = h_Ap[i+1] - h_Ap[i];
        tmap[i] = i;
    }

    thrust::sort_by_key(nnz, nnz + N, tmap, thrust::greater<int>()); // nnz and tmap are both sorted

    int* warp_sizes;
    int num_warps = (N-1)/32 + 1;
    warp_sizes = (int*)malloc(sizeof(int) * num_warps);

    for(int i=0;i<N;i+=32){
        int sz = 0;
        for(int j = 0;j < 32;j++){
            sz += nnz[i+j];
        }
        warp_sizes[i/32] = sz;
    }
    // the last warp won't be fully filled
    if(N%32 != 0){
        int x = (N/32) * 32;
        int sz = 0;
        for(;x < N; x++){
            sz += nnz[x];
        }
         warp_sizes[N/32 + 1] = sz;
    }
    
    int* warp_base = (int*)malloc(sizeof(int) * num_warps);
    thrust::exclusive_scan(warp_sizes, warp_sizes + num_warps, warp_base);

    ValueT *d_nAx;
    int* d_nAj, *d_tmap, *d_nnz, *d_warp_base;
    
    cudaMalloc(&d_nAx, nnz_tot * sizeof(ValueT));
    cudaMalloc(&d_nAj, nnz_tot * sizeof(int));
    cudaMalloc(&d_tmap, N * sizeof(int));
    cudaMalloc(&d_nnz, N * sizeof(int));
    cudaMalloc(&d_warp_base, num_warps * sizeof(int));

    // cudaMemcpy(d_new_ind, new_ind, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmap, tmap, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnz, nnz, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_warp_base, warp_base, num_warps * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (N-1)/512 + 1;
    int num_threads = 512;

     // implementing a memory check...was any index in d_new_ind accessed more/less than once
    int* check = (int*)malloc(sizeof(int) * nnz_tot);
    int* d_check;
    cudaMalloc(&d_check, sizeof(int) * nnz_tot);
    cudaMemset(d_check, 0, sizeof(int) * nnz_tot);
   
    populate<<<num_blocks, num_threads>>>(N, d_Ap, d_Ax, d_Aj, d_nAx, d_nAj, d_tmap, d_nnz, d_warp_base, d_check);
    

    cudaMemcpy(check, d_check, nnz_tot * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<nnz_tot;i++){
        if(check[i]!=1){
            cout<<"fail"<<endl;
            return;
        }
            // cout<<i<<" "<<check[i]<<"\n";
    }

    cout<<"successful xformation"<<endl;

    // int m = N;





	int nthreads = 512;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	spmv_csr_scalar <<<nblocks, nthreads>>> (m, d_Ap, d_nAj, d_nAx, d_x, d_y, d_tmap, d_warp_base, d_nnz);   
	// CudaTest("solving spmv_base kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	double time = t.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz_tot);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz_tot / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, m * sizeof(ValueT), cudaMemcpyDeviceToHost));
	// double error = l2_error(m, y_copy, h_y);
	printf("\truntime [cuda_base] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error]\n", time, GFLOPs, GBYTEs);

	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}