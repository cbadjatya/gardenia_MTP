// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "timer.h"
#include "bfs.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#define BFS_VARIANT "topo_bitonic"



// OPT KERNELS BEGIN

__global__ void preprocess1(int N, int *P, bool *isBad, int *CSR_N, int *numBadWarps, int magic_val)
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
        if (tid % 32 == 0 && max_value - min_value > magic_val) // value based on heuristics!
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


__global__ void bfs_step(int m, int *row_offsets, const IndexT *column_indices, int *front, int *depths, int depth, int* tmap) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) src = tmap[src];
	if(src < m && front[src]) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (depths[dst] == MYINFINITY) {
				atomicMin(&depths[dst], depth);
				// depths[dst] = depth;
			}
		}
	}
}

__global__ void update(int m, int *depths, bool *visited, int *front, bool *changed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(depths[id] != MYINFINITY && !visited[id]) {
			visited[id] = true;
			front[id] = 1;
			*changed = true;
		}
	}
}

void BFSSolver(Graph &g, int source, DistT* h_dist, int magic) {
	//print_device_info(0);
	auto m = g.V();
	auto nnz = g.E();
	auto h_row_offsets = g.out_rowptr();
	auto h_column_indices = g.out_colidx();
	int zero = 0;
	bool one = 1;
	int *d_row_offsets;
	VertexId *d_column_indices;

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

	int * d_dist;	
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(int), cudaMemcpyHostToDevice));
	bool *d_changed, h_changed, *d_visited;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_visited, 0, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(&d_visited[source], &one, sizeof(bool), cudaMemcpyHostToDevice));
	int *d_front;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_front, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_front, 0, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(&d_front[source], &one, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	int iter = 0;
	int nthreads = 512;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA BFS solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);
	
	float milliseconds = 0;

	cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



	cudaEventRecord(start);

    int *P, *d_P, numBadWarps, *d_numBadWarps;
    bool *isBad, *d_isBad;
    int* thread_mappings;
    int *d_Good, *d_Bad, *d_Gi, *d_Bi;
    int Gi, Bi;
    int totalWarps = (m / 32);
    P = (int *)malloc(totalWarps * sizeof(int));
    isBad = (bool *)malloc(totalWarps);
    cudaMalloc(&thread_mappings, m*sizeof(int));
    numBadWarps = 0;
    cudaMalloc(&d_numBadWarps, sizeof(int));
    cudaMemset(d_numBadWarps, 0, sizeof(int));
    cudaMalloc(&d_Gi, sizeof(int));
    cudaMalloc(&d_Bi, sizeof(int));
    cudaMalloc(&d_P, totalWarps * sizeof(int));

    cudaMalloc(&d_isBad, totalWarps);
    cudaMemset(d_isBad, 0, totalWarps);

    cudaMemset(d_numBadWarps, 0, sizeof(int));
    preprocess1<<<ceil(((float)totalWarps * 32) / 512), 512>>>(totalWarps * 32, d_P, d_isBad, d_row_offsets, d_numBadWarps, 350);
    CUDA_SAFE_CALL(cudaMemcpy(&numBadWarps, d_numBadWarps, sizeof(int), cudaMemcpyDeviceToHost));
    cudaMalloc(&d_Good, numBadWarps * sizeof(int));
    cudaMalloc(&d_Bad, numBadWarps * sizeof(int));
    cudaMemset(d_Gi, 0, sizeof(int));
    cudaMemset(d_Bi, 0, sizeof(int));
    preprocess2<<<ceil(totalWarps * 1.0 / 512), 512>>>(d_isBad, numBadWarps, d_Good, d_Bad, totalWarps, d_Gi, d_Bi);
    CUDA_SAFE_CALL(cudaMemcpy(&Gi, d_Gi, sizeof(int), cudaMemcpyDeviceToHost));
    preprocess2_1<<<max(1, (int)ceil(Gi * 1.0 / 512)), 512>>>(d_P, Gi, d_Good, d_Bad);


    preprocess3<<<nblocks, nthreads>>>(m, numBadWarps, d_row_offsets, thread_mappings, d_P, totalWarps);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("opt time = %f\n", milliseconds);



	cudaEventRecord(start);
	// std::cout<<"All good 0 after stop 123456"<<endl;
	do {
		++iter;
		h_changed = false;
		// std::cout<<"All good 1 after stop 123456"<<endl;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		// std::cout<<"All good 2 after stop 123456"<<endl;
		bfs_step <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_front, d_dist, iter, thread_mappings);
		// std::cout<<"All good 3 after stop 123456"<<endl;
		// CudaTest("solving bfs_step failed");
		// std::cout<<"All good 4 after stop 123456"<<endl;
		CUDA_SAFE_CALL(cudaMemset(d_front, 0, m * sizeof(int)));
		// std::cout<<"All good 5 after stop 123456"<<endl;
		update <<<nblocks, nthreads>>> (m, d_dist, d_visited, d_front, d_changed);
		// CudaTest("solving update failed");
		// std::cout<<"All good 6 after stop 123456"<<endl;
		
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		// std::cout<<"All good 7 after stop 123456"<<endl;
		
	} while (h_changed);
	// std::cout<<"All good after stop2"<<endl;
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
	
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n\n\n", BFS_VARIANT, elapsed);
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	CUDA_SAFE_CALL(cudaFree(d_front));
	CUDA_SAFE_CALL(cudaFree(d_changed));
	return;
}
