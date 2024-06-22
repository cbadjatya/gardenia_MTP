// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "atomic_free"
#include "bfs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

__global__ void initialize(int m, int source, int* depth) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(id == source) depth[source] = 0;
		else depth[id] = MYINFINITY;
	}
}

__global__ void bfs_kernel(int m, int *row_offsets, int *column_indices, DistT *dist, bool *changed, int depth) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src >= m) return;

	if(dist[src] == depth - 1) { // visited but not expanded
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];

		for (int offset = row_offsets[src]; offset < row_offsets[src + 1]; ++ offset) {
			int dst = column_indices[offset];
			if (dist[dst] > depth) {
				dist[dst] = depth;
				*changed = true;
			}
		}
	}
}

__global__ void bfs_update(int m, DistT *dist, bool *visited) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(dist[id] < MYINFINITY && !visited[id])
			visited[id] = true;
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

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

	bool *d_changed, h_changed, *d_visited, *d_expanded;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	// CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(bool)));
	// CUDA_SAFE_CALL(cudaMalloc((void **)&d_expanded, m * sizeof(bool)));

	DistT * d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	
	//CUDA_SAFE_CALL(cudaMemset(d_visited, 0, m * sizeof(bool)));
	//CUDA_SAFE_CALL(cudaMemset(d_expanded, 0, m * sizeof(bool)));
	// int *d_num_frontier;
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_frontier, sizeof(int)));

	int iter = 0;
	int nthreads = 512;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, source, d_dist);
	// CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(DistT), cudaMemcpyHostToDevice));
	//int h_num_frontier = 1;
	// cout<<h_row_offsets[source]<<" "<<h_row_offsets[source+1]<<"\n";
	cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaEventRecord(start);
	do {
		++iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy(d_num_frontier, &zero, sizeof(int), cudaMemcpyHostToDevice));
		bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_dist, d_changed, iter);
		// bfs_update <<<nblocks, nthreads>>> (m, d_dist, d_visited);
		// CudaTest("solving failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL(cudaMemcpy(&h_num_frontier, d_num_frontier, sizeof(int), cudaMemcpyDeviceToHost));
		//printf("iteration %d: num_frontier = %d\n", iter, h_num_frontier);
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	//t.Stop();
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, elapsed);

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	// CUDA_SAFE_CALL(cudaFree(d_changed));
	
	return;
}
