#define SSSP_VARIANT "topology"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"

__global__ void initialize(unsigned *dist, unsigned m) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

__global__ void sssp_kernel(int m, int *row_offsets, int *column_indices, W_TYPE *weight, unsigned *dist, bool *changed) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m && dist[src] < MYINFINITY) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1];
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				unsigned altdist = dist[src] + (unsigned)weight[offset];
				if (altdist < dist[dst]) {
					unsigned olddist = atomicMin(&dist[dst], altdist);
					if (altdist < olddist) {
						*changed = true;
					}
				}
			}
		}
	}
}

#define SWAP(a, b)	{ tmp = a; a = b; b = tmp; }

void sssp(int m, int nnz, int *d_row_offsets, int *d_column_indices, W_TYPE *d_weight, unsigned *d_dist) {
	unsigned zero = 0;
	bool *d_changed, h_changed;
	double starttime, endtime, runtime;
	int iteration = 0;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	initialize <<<nblocks, nthreads>>> (d_dist, m);
	CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[0], &zero, sizeof(zero), cudaMemcpyHostToDevice));

	const size_t max_blocks = maximum_residency(sssp_kernel, nthreads, 0);
	//const size_t max_blocks = 6;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++iteration;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
		printf("iteration=%d\n", iteration);
		sssp_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_weight, d_dist, d_changed);
		CudaTest("solving failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_changed));
	return;
}
