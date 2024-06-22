// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>

/*
Gardenia Benchmark Suite
Kernel: Stochastic Gradient Descent (SGD)
Author: Xuhao Chen
*/
#define SGD_VARIANT "opt"
#define THRUST_IGNORE_CUB_VERSION_CHECK
#include "sgd.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;


// OPT KERNELS BEGIN

__global__ void preprocess1(int N, int *P, bool *isBad, int *CSR_N, int *numBadWarps)
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
        if (tid % 32 == 0 && max_value - min_value > 1000) // value based on heuristics!
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

__global__ void preprocess3(int N, int numBadWarps, int* d_offset, int* thread_map, int* P, int totalWarps){

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



__global__ void update(int m, int n, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT lambda, ScoreT step, ScoreT *squared_errors,
                            const int* tmap) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < m) {
		//int user_id = ordering[tid];
        tid = tmap[tid];
		int user_id = tid;
		int row_begin = row_offsets[user_id];
		int row_end = row_offsets[user_id+1]; 
		int user_offset = K * user_id;
		LatentT *ulv = &user_lv[user_offset];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int item_id = column_indices[offset];
			int item_offset = K * item_id;
			LatentT *ilv = &item_lv[item_offset];
			ScoreT estimate = 0;
			for (int i = 0; i < K; i++)
				estimate += ulv[i] * ilv[i];
			ScoreT delta = rating[offset] - estimate;
			squared_errors[user_id] += delta * delta;
			for (int i = 0; i < K; i++) {
				LatentT p_u = ulv[i];
				LatentT p_i = ilv[i];
				ulv[i] += step * (-lambda * p_u + p_i * delta);
				ilv[i] += step * (-lambda * p_i + p_u * delta);
			}
		}
	}
}

__global__ void rmse(int m, ScoreT *squared_errors, ScoreT *total_error) {
	int uid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	ScoreT local_error = 0.0;
	if(uid < m) local_error = squared_errors[uid];
	ScoreT block_sum = BlockReduce(temp_storage).Sum(local_error);
	if(threadIdx.x == 0) atomicAdd(total_error, block_sum);
}

void SGDSolver(int num_users, int num_items, int nnz, int *h_row_offsets, int *h_column_indices, 
                                            ScoreT *h_rating, LatentT *h_user_lv, LatentT *h_item_lv, int *h_ordering) {
	//print_device_info(0);
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (num_users + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (num_users + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_rating;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_rating, nnz * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_rating, h_rating, nnz * sizeof(ScoreT), cudaMemcpyHostToDevice));
	// int *d_ordering;
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_ordering, num_users * sizeof(int)));
	//CUDA_SAFE_CALL(cudaMemcpy(d_ordering, h_ordering, num_users * sizeof(int), cudaMemcpyHostToDevice));

	LatentT *d_user_lv, *d_item_lv;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_user_lv, num_users * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_item_lv, num_items * K * sizeof(LatentT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_user_lv, h_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_item_lv, h_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyHostToDevice));
	ScoreT h_error, *d_error, *squared_errors;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_error, sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&squared_errors, num_users * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemset(d_error, 0, sizeof(ScoreT)));

	int iter = 0;
	int nthreads = 512;
	int nblocks = (num_users - 1) / nthreads + 1;
	printf("Launching CUDA SGD solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	Timer t1;
	t1.Start();

    // Adding OPT Code

    int *P, *d_P, numBadWarps, *d_numBadWarps;
    bool *isBad, *d_isBad;
    int totalWarps = (num_users / 32);
    P = (int *)malloc(totalWarps * sizeof(int));
    isBad = (bool *)malloc(totalWarps);

    numBadWarps = 0;
    cudaMalloc(&d_numBadWarps, sizeof(int));
    cudaMemset(d_numBadWarps, 0, sizeof(int));

    cudaMalloc(&d_P, totalWarps * sizeof(int));

    cudaMalloc(&d_isBad, totalWarps);
    cudaMemset(d_isBad, 0, totalWarps);

    preprocess1<<<ceil(((float)totalWarps * 32) / 512), 512>>>(totalWarps * 32, d_P, d_isBad, d_row_offsets, d_numBadWarps);
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
    cudaMalloc(&thread_mappings, num_users*sizeof(int));
    preprocess3<<<nblocks, nthreads>>>(num_users, numBadWarps, d_row_offsets, thread_mappings, d_P, totalWarps);

    // OPT code ends
    cudaDeviceSynchronize();
    t1.Stop();
    printf("\truntime for opt [%s] = %f ms.\n", SGD_VARIANT, t1.Millisecs());

    Timer t;
    t.Start();

	do {
		++iter;
		h_error = 0.0;
		CUDA_SAFE_CALL(cudaMemset(squared_errors, 0, num_users * sizeof(ScoreT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_error, &h_error, sizeof(ScoreT), cudaMemcpyHostToDevice));
		update<<<nblocks, nthreads>>>(num_users, num_items, d_row_offsets, d_column_indices, d_rating, d_user_lv, d_item_lv, lambda, step, squared_errors, thread_mappings);
		// CudaTest("solving kernel update failed");
		rmse<<<nblocks, nthreads>>>(num_users, squared_errors, d_error);
		// CudaTest("solving kernel rmse failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(ScoreT), cudaMemcpyDeviceToHost));
		//printf("h_error=%f\n", h_error);
		printf("iteration %d: RMSE error = %f\n", iter, sqrt(h_error/nnz));
		//CUDA_SAFE_CALL(cudaMemcpy(h_user_lv, d_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL(cudaMemcpy(h_item_lv, d_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
		//print_latent_vector(num_users, num_items, h_user_lv, h_item_lv);
	} while (iter < max_iters && h_error > epsilon);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_user_lv, d_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_item_lv, d_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_rating));
	CUDA_SAFE_CALL(cudaFree(d_user_lv));
	CUDA_SAFE_CALL(cudaFree(d_item_lv));
	CUDA_SAFE_CALL(cudaFree(d_error));
	CUDA_SAFE_CALL(cudaFree(squared_errors));
}

