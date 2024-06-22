// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "timer.h"
#include "bfs.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <thrust/sort.h>
#include <thrust/scan.h>
#define BFS_VARIANT "topo_swell"


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


__global__ void populate(int N, int* d_row_offsets, int* Aj ,int* nAj, int* tmap, int* nnz, int* warp_base, int* check, int* rev_map){

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N) return;


    int nid = tmap[id]; // nid is the vertex this thread is supposed to process
    int wid = id / 32;
    int wsz = min(32, N - wid*32);

    int wbase = warp_base[wid] + id % wsz; // the base address for this vertex
   
    int row_begin = d_row_offsets[nid];
    int row_end = d_row_offsets[nid + 1];

    for(int i = row_begin; i<row_end; i++){
    
        nAj[wbase] = rev_map[Aj[i]];
        // nAx[wbase] = Ax[i];

        atomicAdd(&check[wbase],1);
        
        wbase += wsz;
    }

    // this mapping might be called ELL-32-C (or some such thing where 32 is sigma)

}


__global__ void bfs_step(int N, int *row_offsets, const int *column_indices, int *front, int *depths, int depth, int* tmap, int* warp_base, int* nnz) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(src < N && front[src]) {
        int wid = src/32;
        int wsz = min(32, N - wid*32);
        int wbase = warp_base[wid] + src%wsz;
		int ncols = nnz[src];
		for (int offset = 0; offset < ncols; ++ offset) {
             int index = wbase + wsz*offset;
			int dst = column_indices[index];
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
	auto nnz_tot = g.E();
	auto h_row_offsets = g.out_rowptr();
	auto h_column_indices = g.out_colidx();
	int zero = 0;
	bool one = 1;
	int *d_row_offsets;
	VertexId *d_column_indices;

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz_tot * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz_tot * sizeof(VertexId), cudaMemcpyHostToDevice));

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

    int N = m;

	cudaEventRecord(start);

    int* nnz = (int*)malloc(N * sizeof(int)); // number of non zeroes (no. of neighbors for each node)
    
    int* tmap = (int*)malloc(N * sizeof(int)); // remapping id for each vertex

    for(int i=0;i<N;i++){
        nnz[i] = h_row_offsets[i+1] - h_row_offsets[i];
        tmap[i] = i;
    }
    int* rev_map = (int*)malloc(N * sizeof(int));

    thrust::sort_by_key(nnz, nnz + N, tmap, thrust::greater<int>()); // nnz and tmap are both sorted


    for(int i=0;i<N;i++){
        rev_map[tmap[i]] = i;
    }


    int* warp_sizes;
    int num_warps = (N-1)/32 + 1;
    warp_sizes = (int*)malloc(sizeof(int) * num_warps);
    int tot_size = 0;

    for(int i=0;i<N;i+=32){
        warp_sizes[i/32] = nnz[i]*32;
        tot_size += nnz[i]*32;
    }
    // the last warp won't be fully filled
    if(N%32 != 0){
        int x = (N/32) * 32;
        
         warp_sizes[N/32 + 1] = nnz[x]*(N-x);
         tot_size += nnz[x]*(N-x);
    }
    
    int* warp_base = (int*)malloc(sizeof(int) * num_warps);
    thrust::exclusive_scan(warp_sizes, warp_sizes + num_warps, warp_base);

    int *d_nAj , *d_tmap, *d_nnz, *d_warp_base, *d_rev_map;

    cudaMalloc(&d_nAj, tot_size * sizeof(int));

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
   
    populate<<<num_blocks, num_threads>>>(N, d_row_offsets, d_column_indices, d_nAj, d_tmap, d_nnz, d_warp_base, d_check, d_rev_map);
    

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
    printf("opt time = %f\n", milliseconds);



	cudaEventRecord(start);
	// std::cout<<"All good 0 after stop 123456"<<endl;
	do {
		++iter;
		h_changed = false;
		// std::cout<<"All good 1 after stop 123456"<<endl;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		// std::cout<<"All good 2 after stop 123456"<<endl;
		bfs_step <<<nblocks, nthreads>>> (m, d_row_offsets, d_nAj, d_front, d_dist, iter, d_tmap, d_warp_base, d_nnz);
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

    int* h_dist_rev = (int*)malloc(m * sizeof(int));
    for(int i=0;i<m;i++){
        h_dist_rev[i] = h_dist[tmap[i]];
    }
    h_dist = h_dist_rev;
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	CUDA_SAFE_CALL(cudaFree(d_front));
	CUDA_SAFE_CALL(cudaFree(d_changed));
	return;
}
