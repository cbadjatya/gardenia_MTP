// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "spmv.h"
#include "timer.h"
#include "spmv_util.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"



// OPT KERNELS BEGIN

__global__ void preprocess1(int N, int *P, bool *isBad, uint64_t *CSR_N, int *numBadWarps)
{
    // Set the warp reference array P. count Bad Warps and mark each bad warp in the array isBad.
    // N here is the largest multiple of 32 <= actual N.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
    {
        int wid = tid / 32;
        if (tid % 32 == 0)
            P[wid] = wid; // set the initial reference
        int max_value = (CSR_N[tid + 1] - CSR_N[tid]);
        int min_value = max_value;
        for (int i = 16; i > 0; i = i / 2)
        {
            max_value = max(max_value, __shfl_down_sync(-1, max_value, i));
            min_value = min(min_value, __shfl_down_sync(-1, min_value, i));
        }
        if (tid % 32 == 0 && max_value - min_value > 700) // value based on heuristics!
        {
            isBad[wid] = true;
            atomicAdd(numBadWarps, 1);
        }
    }
}

__global__ void preprocess2(bool *isBad, int badWarps, int *G, int *B, int totalWarps, int *Gi, int *Bi)
{
    // kernel launched with totalWarps number of threads

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool flag;
    if (tid < totalWarps)
    {
        flag = isBad[tid];
        if (tid < badWarps)
        {
            if (!flag)
            {
                int i = atomicAdd(Gi, 1);
                G[i] = tid;
            }
        }
        else if (flag)
        {
            int i = atomicAdd(Bi, 1);
            B[i] = tid;
        }
    }
    // launch a dynamic kernel here?
    
}

__global__ void preprocess2_1(int *P, int Gi, int *G, int *B)
{
    // kernel launched with numBadWarps number of threads.

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < Gi)
    {
        int temp1 = G[tid];
        int temp2 = B[tid];
        P[temp1] = temp2;
        P[temp2] = temp1;
    }
}

__global__ void preprocess3(int N, int numBadWarps, uint64_t* d_offset, int* thread_map, int* P, int totalWarps){

   __shared__ int ind[512][2];

    unsigned int i, ij, v, wid_orig, wid_new;

    i = threadIdx.x;
    v = i + blockIdx.x * blockDim.x;
    int id = v;

    if (v >= N) return;

    thread_map[v] = v;

    wid_orig = v / 32;
    wid_new = wid_orig;


    if (wid_new < totalWarps){
        wid_new = P[wid_new];
        v = wid_new * 32 + i % 32; // new id according to new warp arrangement
        thread_map[id] = v;
    }

    if ((blockIdx.x * blockDim.x) / 32 < numBadWarps)
    {

        ind[threadIdx.x][0] = (d_offset[v + 1] - d_offset[v]); // the loop's limit
        ind[threadIdx.x][1] = v;

        // if(i == 0) atomicAdd(oblocks,1); // DEBUG STUFF

        for (int k = 2; k <= 512; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {

                ij = i ^ j;

                if (ij > i)
                {
                    int temp[2];
                    if (((i & k) == 0 && ind[i][0] > ind[ij][0]) || ((i & k) != 0 && ind[i][0] < ind[ij][0]))
                    {
                        temp[0] = ind[i][0];
                        temp[1] = ind[i][1];
                        ind[i][0] = ind[ij][0];
                        ind[i][1] = ind[ij][1];
                        ind[ij][0] = temp[0];
                        ind[ij][1] = temp[1];
                    }
                }
                __syncthreads();
            }
        }

        thread_map[id] = ind[threadIdx.x][1];
    }


}

// OPT KERNELS END




// CSR SpMV kernels based on a scalar model (one thread per row)
// Straightforward translation of standard CSR SpMV to CUDA
// where each thread computes y[i] += A[i,:] * x 
// (the dot product of the i-th row of A with the x vector)
__global__ void spmv_csr_scalar(int m, const uint64_t* Ap, 
                                const VertexId* Aj, const ValueT * Ax, 
                                const ValueT * x, ValueT * y, int* tmap) {
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m) {
		row = tmap[row];
		ValueT sum = y[row];
		unsigned int row_begin = Ap[row];
		unsigned int row_end = Ap[row+1];
		for (unsigned int offset = row_begin; offset < row_end; offset ++){
			sum += Ax[offset] * x[Aj[offset]];
		}
		y[row] = sum;
	}
}

void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) {
  auto m = g.V();
  auto nnz = g.E();
	auto h_Ap = g.in_rowptr();
	auto h_Aj = g.in_colidx();	
	//print_device_info(0);
	uint64_t *d_Ap;
  VertexId *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

	ValueT *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
	for(int i = 0; i < m; i ++) y_copy[i] = h_y[i];
	// SpmvSerial(m, nnz, h_Ap, h_Aj, h_Ax, h_x, y_copy);
	
	int nthreads = 512;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();



    // Adding OPT Code

    int *P, *d_P, numBadWarps, *d_numBadWarps;
    bool *isBad, *d_isBad;
    int totalWarps = (m / 32);
    P = (int *)malloc(totalWarps * sizeof(int));
    isBad = (bool *)malloc(totalWarps);

    numBadWarps = 0;
    cudaMalloc(&d_numBadWarps, sizeof(int));
    cudaMemset(d_numBadWarps, 0, sizeof(int));

    cudaMalloc(&d_P, totalWarps * sizeof(int));

    cudaMalloc(&d_isBad, totalWarps);
    cudaMemset(d_isBad, 0, totalWarps);

    preprocess1<<<ceil(((float)totalWarps * 32) / 512), 512>>>(totalWarps * 32, d_P, d_isBad, d_Ap, d_numBadWarps);
    // CHECK_DEBUG(cudaMemcpy(P,d_P,totalWarps*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&numBadWarps, d_numBadWarps, sizeof(int), cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(isBad,d_isBad,totalWarps,cudaMemcpyDeviceToHost));

    int *d_Good, *d_Bad, *d_Gi, *d_Bi;
    int Gi, Bi;

    cudaMalloc(&d_Good, numBadWarps * sizeof(int));
    cudaMalloc(&d_Bad, numBadWarps * sizeof(int));
    cudaMalloc(&d_Gi, sizeof(int));
    cudaMalloc(&d_Bi, sizeof(int));
    cudaMemset(d_Gi, 0, sizeof(int));
    cudaMemset(d_Bi, 0, sizeof(int));

    preprocess2<<<ceil(totalWarps * 1.0 / 512), 512>>>(d_isBad, numBadWarps, d_Good, d_Bad, totalWarps, d_Gi, d_Bi);

    CUDA_SAFE_CALL(cudaMemcpy(&Gi, d_Gi, sizeof(int), cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(&Bi, d_Bi,sizeof(int), cudaMemcpyDeviceToHost));

    // cudaFree(d_isBad);

    preprocess2_1<<<max(1, (int)ceil(Gi * 1.0 / 512)), 512>>>(d_P, Gi, d_Good, d_Bad);

    // cudaFree(d_Good);
    // cudaFree(d_Bad);

    // CHECK(cudaMemcpy(P,d_P,totalWarps*sizeof(int),cudaMemcpyDeviceToHost));
    int* thread_mappings;
    cudaMalloc(&thread_mappings, m*sizeof(int));
    preprocess3<<<nblocks, nthreads>>>(m, numBadWarps, d_Ap, thread_mappings, d_P, totalWarps);
    cudaDeviceSynchronize();
    // OPT code ends
    t.Stop();
    printf("time taken for bitonic opt = %f\n",t.Millisecs());

    Timer t1;
    t1.Start();



	spmv_csr_scalar <<<nblocks, nthreads>>> (m, d_Ap, d_Aj, d_Ax, d_x, d_y, thread_mappings);   
	// CudaTest("solving spmv_base kernel failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t1.Stop();

	double time = t1.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, m * sizeof(ValueT), cudaMemcpyDeviceToHost));
	double error = l2_error(m, y_copy, h_y);
	printf("\truntime [cuda_base] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n\n\n", time, GFLOPs, GBYTEs, error);

	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}

