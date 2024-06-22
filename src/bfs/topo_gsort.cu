// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "timer.h"
#include "bfs.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#define BFS_VARIANT "topo_gsort"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>


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
        if (tid % 32 == 0){
            if(max_value - min_value > magic_val){
                isBad[wid] = true;
                atomicAdd(numBadWarps, 1);
            }
            else{
                isBad[wid] = false;
            }
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

   __shared__ unsigned ind[512][2];

    unsigned int i, ij, v, wid_new;

    i = threadIdx.x;
    v = i + blockIdx.x * blockDim.x;
    unsigned int id = v;

    if (v >= N) return;

    thread_map[v] = v;
    wid_new = v / 32;

    if (wid_new < totalWarps){
        wid_new = P[wid_new];
        v = wid_new * 32 + i % 32; // new id according to new warp arrangement
        thread_map[id] = v;
    }


    return;



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

// keeping grp_cnt global right now, planning to do a complete sort (for bad blocks)

// for each warp, split into groups based on workload of the constituting threads.
// The sort will then be applied on all groups. 
// do I need to use newId for indexing anything except offsets?

// do global variables need to be made volatile? 

__global__ void preprocess4(int* thread_map, int* d_offsets, unsigned int* grp_cnt, int* threadToGroup, int* localIndexInGroup, int* group_map, int* group_size, int* group_load, int numBadWarps
                                , int GROUP_THRESHOLD){

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id >= numBadWarps* 32) return;

    int node_to_be_processed = thread_map[id];


    if(threadIdx.x % 32 == 0){
        
        int curr_sz = 1;
        int curr_load = d_offsets[node_to_be_processed+1] - d_offsets[node_to_be_processed];
        int curr_groupId = atomicInc(grp_cnt, INT_MAX);
        threadToGroup[id] = curr_groupId;
        localIndexInGroup[id] = 0;

        for(int i = 1; i < 32; i++){

            int load_i = d_offsets[node_to_be_processed + 1 + i] - d_offsets[node_to_be_processed + i];

            if(curr_load > 32 || curr_load + load_i > GROUP_THRESHOLD){
                group_map[curr_groupId] = curr_groupId; // do we need this??
                group_size[curr_groupId] = curr_sz;
                group_load[curr_groupId] = curr_load;

                curr_load = 0;
                curr_groupId = atomicInc(grp_cnt, INT_MAX);
                curr_sz = 0;
            }

            curr_load += load_i;
            curr_sz++;
            threadToGroup[id + i] = curr_groupId; // using new_Id here. Correct since local mapping for threads within a warp is still the same.
            localIndexInGroup[id + i] = curr_sz - 1;

        }

        group_map[curr_groupId] = curr_groupId;
        group_size[curr_groupId] = curr_sz;
        group_load[curr_groupId] = curr_load;
    }
    
}

// do I need to sort groupMap or simply sort threadToGroupMapping based on the size...but would it be a key value pair mapping??

// calculate a prefix sum with the sizes of each group pref[grp]

// once all the groups have been sorted, id needs to be reassigned from thread_map[id] to 
// pref[group_map[threadToGroup[thread_map[id]]]] - group_size[group_map[threadToGroup[thread_map[id]]]] + localIndexInGroup[new_id]
// do I actually need to sort group_size too??

__global__ void preprocess5(int* thread_map, int numBadWarps, int* threadToGroup, int* r_map, int* pref, int* group_size, int* localIndexInGroup){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if( id >= numBadWarps* 32) return;

    int old_grp = threadToGroup[id];
    int new_group = r_map[old_grp];
    int grpPos = pref[new_group] - group_size[new_group]; // sizes need to be sorted to calculate prefix sum...
    
    thread_map[grpPos + localIndexInGroup[id]] = thread_map[id];

}

__global__ void make_rev_map(int* rmap, int* gmap, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N){
        rmap[gmap[id]] = id;
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
	
	cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	float milliseconds = 0;

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

    int* hmap;
    int rmpd = numBadWarps*32;

    hmap = (int*)malloc(sizeof(int) * (rmpd + 1));
    cudaMemcpy(hmap, thread_mappings, (rmpd+1)* sizeof(int), cudaMemcpyDeviceToHost);

    unsigned int* grp_cnt;
    int* threadToGroup;
    int* localIndexInGroup;
    int* group_map;
    int* group_size;
    int* group_load;
    int* pref;


    cudaMalloc(&grp_cnt, sizeof(int));
    cudaMalloc(&threadToGroup, numBadWarps*32*sizeof(int));
    cudaMalloc(&localIndexInGroup, numBadWarps*32*sizeof(int));
    cudaMalloc(&group_map, numBadWarps*32*sizeof(int));
    cudaMalloc(&group_size, numBadWarps*32*sizeof(int));
    cudaMalloc(&group_load, numBadWarps*32*sizeof(int));


    cudaMemset(grp_cnt, 0, sizeof(int));

    

    int p4block_size = 512;
    int p4blocks = (numBadWarps*32 - 1)/512 + 1;

    preprocess4<<<p4blocks, p4block_size>>>(thread_mappings, d_row_offsets, grp_cnt, threadToGroup, localIndexInGroup, group_map, group_size, group_load, numBadWarps, 50);


    int h_grp_cnt;
    cudaMemcpy(&h_grp_cnt, grp_cnt, sizeof(int), cudaMemcpyDeviceToHost);

    auto zipIt = thrust::make_zip_iterator(thrust::make_tuple(group_map, group_size));

    thrust::sort_by_key(thrust::device, group_load, group_load+h_grp_cnt, zipIt);

    cudaMalloc(&pref, h_grp_cnt*sizeof(int));
    cudaMemcpy(pref, group_size, h_grp_cnt * sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::inclusive_scan(thrust::device, pref, pref+h_grp_cnt, pref);

    // make reverse group mapping
    int* r_map;
    cudaMalloc(&r_map, sizeof(int)* h_grp_cnt);
    make_rev_map<<<(h_grp_cnt - 1)/512 + 1, 512>>>(r_map, group_map, h_grp_cnt);

    preprocess5<<<p4blocks, p4block_size>>>(thread_mappings, numBadWarps, threadToGroup, r_map, pref, group_size, localIndexInGroup);

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
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, elapsed);
	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
	CUDA_SAFE_CALL(cudaFree(d_front));
	CUDA_SAFE_CALL(cudaFree(d_changed));
	return;
}
