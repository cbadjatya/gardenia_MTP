#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "bfs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"


__global__ void bfs_step(int N, int *row_offsets, int *column_indices, int *front, int *depths, int depth, int* tmap, int* warp_base, int* nnz) {
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
    if(0 == nnz[id])
        atomicAdd(&sh_offsets[lwid], 1);
    //  __syncwarp();

    // check if any one node is in the frontier, if yes every thread runs
    int active = front[nid];
    int w_active = active;

    for (int offset = 16; offset > 0; offset /= 2)
        w_active = __shfl_down_sync(-1, w_active, offset) | w_active;

    if(w_active == 0) return;
    
	int row_begin = row_offsets[nid];
    int row_end = row_offsets[nid+1];
    for(int i = 0;i < nnz[id];){
        if(active){
            int dst = column_indices[wbase];
            if (depths[dst] == MYINFINITY) {
                atomicMin(&depths[dst], depth);
            }
        }

        wbase -= sh_offsets[lwid];
        i++;
        if(i == nnz[id])
            atomicAdd(&sh_offsets[lwid], 1);
        // __syncwarp();
        wbase += wsz;
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

__global__ void populate(int N, int* row_offsets, int* column_indices, int* new_ind, int* tmap, int* nnz, int* warp_base, int* d_check){
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
        int ind = column_indices[i + row_offsets[nid]];
        new_ind[wbase] = ind;
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


void BFSSolver(Graph &g, int source, int* h_dist, int magic) {
	//print_device_info(0);
	auto N = g.V();
	auto F_sz = g.E();
	auto h_row_offsets = g.out_rowptr();
	auto h_column_indices = g.out_colidx();

    int* nnz;
    int* tmap;

    nnz = (int*)malloc(N * sizeof(int));
    
    tmap = (int*)malloc(N * sizeof(int));

    for(int i=0;i<N;i++){
        nnz[i] = h_row_offsets[i+1] - h_row_offsets[i];
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

    int* d_row_offsets, *d_column_indices, *d_new_ind, *d_tmap, *d_nnz, *d_warp_base;
    cudaMalloc(&d_row_offsets, (N+1) * sizeof(int));
    cudaMalloc(&d_column_indices, F_sz * sizeof(int));
    cudaMalloc(&d_new_ind, F_sz * sizeof(int));
    cudaMalloc(&d_tmap, N * sizeof(int));
    cudaMalloc(&d_nnz, N * sizeof(int));
    cudaMalloc(&d_warp_base, num_warps * sizeof(int));

    cudaMemcpy(d_row_offsets, h_row_offsets, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, h_column_indices, F_sz * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_ind, new_ind, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmap, tmap, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnz, nnz, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_warp_base, warp_base, num_warps * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (N-1)/512 + 1;
    int num_threads = 512;

     // implementing a memory check...was any index in d_new_ind accessed more/less than once
    int* check = (int*)malloc(sizeof(int) * F_sz);
    int* d_check;
    cudaMalloc(&d_check, sizeof(int) * F_sz);
    cudaMemset(d_check, 0, sizeof(int) * F_sz);
    // cout<<"ok"<<endl;
    for(int i=0;i<1;i++){
        populate<<<num_blocks, num_threads>>>(N, d_row_offsets, d_column_indices, d_new_ind, d_tmap, d_nnz, d_warp_base, d_check);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
            printf("%d\n",i);
            for(int j=i*32;j<i*32 + 32; j++){
                cout<<nnz[j]<<endl;
            }
            return;
        }
    }
    

    cudaMemcpy(check, d_check, F_sz * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<F_sz;i++){
        if(check[i]!=1){
            cout<<"fail"<<endl;
            return;
        }
            // cout<<i<<" "<<check[i]<<"\n";
    }

    cout<<"successful xformation"<<endl;

    int m = N;

    int zero = 0;
	bool one = 1;

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
		bfs_step <<<nblocks, nthreads>>> (m, d_row_offsets, d_new_ind, d_front, d_dist, iter, d_tmap, d_warp_base, d_nnz);
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
	printf("\truntime [%s] = %f ms.\n", "swell", elapsed);
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	CUDA_SAFE_CALL(cudaFree(d_front));
	CUDA_SAFE_CALL(cudaFree(d_changed));
	return;
	
	return;
}

