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


#define CHECK(call) \
    {               \
        call;       \
    }

#define CHECK_DEBUG(call)                                                         \
    {                                                                             \
        const cudaError_t error = call;                                           \
        if (error != cudaSuccess)                                                 \
        {                                                                         \
            printf("Error : %s: %d -> ", __FILE__, __LINE__);                     \
            printf("code : %d, reason : %s\n", error, cudaGetErrorString(error)); \
        }                                                                         \
    }

using namespace std;


__global__ void populate(int N, int* Ap, int* Aj, ValueT* Ax ,int* nAj, ValueT* nAx, int* tmap, int* nnz, int* warp_base, int* check, int* rev_map){

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N) return;


    int nid = tmap[id]; // nid is the vertex this thread is supposed to process
    int wid = id / 1024;
    int wsz = min(1024, N - wid*1024);

    int wbase = warp_base[wid] + id % wsz; // the base address for this vertex
   
    int row_begin = Ap[nid];
    int row_end = Ap[nid + 1];

    for(int i = row_begin; i<row_end; i++){
    
        nAj[wbase] = rev_map[Aj[i]];
        nAx[wbase] = Ax[i];

        atomicAdd(&check[wbase],1);
        
        wbase += wsz;
    }

    // this mapping might be called ELL-1024-C (or some such thing where 1024 is sigma)

}

__global__ void spmv_csr_scalar(int N,
                                const int* Aj, const ValueT * Ax, 
                                const ValueT * x, ValueT * y, int* tmap, int* warp_base, int* nnz) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < N) {
        int wid = id/1024;
        int wsz = min(1024, N - wid*1024);
        int wbase = warp_base[wid] + id%1024;
		ValueT sum = y[id];
		int ncols = nnz[id];
		for (int offset = 0; offset < ncols; offset ++){
            int index = wbase + wsz*offset ;
			sum += Ax[index] * x[Aj[index]];
		}
		y[id] = sum;
	}
}

void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) {
  auto N = g.V();
  auto m  = N;
  auto nnz_tot = g.E();
	auto Ap = g.in_rowptr();
	auto Aj = g.in_colidx();	
	//print_device_info(0);
	int *d_Ap;
    int *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz_tot * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, Ap, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, Aj, nnz_tot * sizeof(int), cudaMemcpyHostToDevice));

	ValueT *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz_tot));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz_tot * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	// ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
	// for(int i = 0; i < m; i ++) y_copy[i] = h_y[i];
	// SpmvSerial(m, nnz_tot, Ap, Aj, h_Ax, h_x, y_copy);
    int* nnz = (int*)malloc(N * sizeof(int)); // number of non zeroes (no. of neighbors for each node)
    
    int* tmap = (int*)malloc(N * sizeof(int)); // remapping id for each vertex

    for(int i=0;i<N;i++){
        nnz[i] = Ap[i+1] - Ap[i];
        tmap[i] = i;
    }
    int* rev_map = (int*)malloc(N * sizeof(int));

    thrust::sort_by_key(nnz, nnz + N, tmap, thrust::greater<int>()); // nnz and tmap are both sorted


    for(int i=0;i<N;i++){
        rev_map[tmap[i]] = i;
    }


    int* warp_sizes;
    int num_warps = (N-1)/1024 + 1;
    warp_sizes = (int*)malloc(sizeof(int) * num_warps);
    int tot_size = 0;

    for(int i=0;i<N;i+=1024){
        warp_sizes[i/1024] = nnz[i]*1024;
        tot_size += nnz[i]*1024;
    }
    // the last warp won't be fully filled
    if(N%1024 != 0){
        int x = (N/1024) * 1024;
        
         warp_sizes[N/1024 + 1] = nnz[x]*(N-x);
         tot_size += nnz[x]*(N-x);
    }
    
    int* warp_base = (int*)malloc(sizeof(int) * num_warps);
    thrust::exclusive_scan(warp_sizes, warp_sizes + num_warps, warp_base);

    int *d_nAj , *d_tmap, *d_nnz, *d_warp_base, *d_rev_map;
    ValueT* d_nAx;

    cudaMalloc(&d_nAj, tot_size * sizeof(int));
    cudaMalloc(&d_nAx, tot_size * sizeof(ValueT));

    cudaMalloc(&d_tmap, N * sizeof(int));
    cudaMalloc(&d_rev_map, N * sizeof(int));
    cudaMalloc(&d_nnz, N * sizeof(int));
    cudaMalloc(&d_warp_base, num_warps * sizeof(int));

   
    cudaMemcpy(d_tmap, tmap, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rev_map, rev_map, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnz, nnz, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_warp_base, warp_base, num_warps * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (N-1)/512 + 1;
    int num_threads = 512;

     // implementing a memory check...was any index in d_new_ind accessed more/less than once
    int* check = (int*)malloc(sizeof(int) * tot_size);
    int* d_check;
    cudaMalloc(&d_check, sizeof(int) * tot_size);
    cudaMemset(d_check, 0, sizeof(int) * tot_size);
   
    populate<<<num_blocks, num_threads>>>(N, d_Ap, d_Aj, d_Ax, d_nAj, d_nAx, d_tmap, d_nnz, d_warp_base, d_check, d_rev_map);
    

    CUDA_SAFE_CALL(cudaMemcpy(check, d_check, nnz_tot * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i=0;i<tot_size;i++){
        if(check[i] > 1){
            cout<<"fail"<<endl;
            return;
        }
            // cout<<i<<" "<<check[i]<<"\n";
    }

    cout<<"successful xformation"<<endl;

    // int m = N;

	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", num_blocks, num_threads);

	Timer t;
	t.Start();
	spmv_csr_scalar <<<num_blocks, num_threads>>> (m, d_nAj, d_nAx, d_x, d_y, d_tmap, d_warp_base, d_nnz);   
	// CudaTest("solving spmv_base kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	double time = t.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz_tot);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz_tot / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, m * sizeof(ValueT), cudaMemcpyDeviceToHost));
	// double error = l2_error(m, y_copy, h_y);
	printf("\truntime [cuda_base] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error]\n\n\n", time, GFLOPs, GBYTEs);

	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}