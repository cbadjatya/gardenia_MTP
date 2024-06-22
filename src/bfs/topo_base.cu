// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "timer.h"
#include "bfs.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#define BFS_VARIANT "topo_base"

__global__ void bfs_step(int m, int *row_offsets, const IndexT *column_indices, int *front, int *depths, int depth) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
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
	int *d_column_indices;

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));

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
	
	cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaEventRecord(start);
	// std::cout<<"All good 0 after stop 123456"<<endl;
	do {
		++iter;
		h_changed = false;
		// std::cout<<"All good 1 after stop 123456"<<endl;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		// std::cout<<"All good 2 after stop 123456"<<endl;
		bfs_step <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_front, d_dist, iter);
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
