// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#define BFS_VARIANT "opt"
#include "bfs.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

// OPT KERNELS BEGIN

// custom sort comparator
struct Local {
    Local(uint64_t* paramA) { this->paramA = paramA; }
    int operator () (int i, int j) {
        if(paramA[i+1]-paramA[i] > paramA[j+1]-paramA[j]) return 1;
        return -1;
    }

    uint64_t* paramA;
};

typedef struct Local comp;

__global__ void preprocess1(int N, int *P, bool *isBad, uint64_t *CSR_N, int *numBadWarps, int magic_val)
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

__global__ void preprocess3(int N, int numBadWarps, uint64_t* d_offset, int* thread_map, int* P, int totalWarps){

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

__global__ void preprocess4(){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id >= numBadWarps* 32) return;

    int new_id = thread_map[id];


    if(id%32 == 0){
        
        int curr_sz = 1;
        int curr_load = d_offsets[new_id+1] - d_offsets[new_id];
        int curr_groupId = atomicAdd(&grp_cnt, 1);
        threadToGroup[new_id] = curr_groupId;
        localIndexInGroup[new_id] = 0;

        for(int i = 1; i < 32; i++){

            int load_i = d_offsets[new_id + 1 + i] - d_offsets[new_id + i];

            if(curr_load > 32 || curr_load + load_i > GROUP_THRESHOLD){
                group_map[curr_groupId] = curr_groupId;
                group_size[curr_groupId] = curr_sz;
                group_load[curr_groupId] = curr_load;

                curr_load = 0;
                curr_groupId = atomicAdd(&grp_cnt, 1);
                curr_sz = 0;
            }

            curr_load += load_i;
            curr_sz++;
            threadToGroup[new_id + i] = curr_groupId; // using new_Id here. Correct since local mapping for threads within a warp is still the same.
            localIndexInGroup[new_id + i] = curr_sz - 1;

        }

        group_map[curr_groupId] = curr_groupId;
        group_size[curr_groupId] = curr_sz;
        group_load[curr_groupId] = curr_load;
    }
    
}

// calculate a prefix sum with the sizes of each group pref[grp]

// once all the groups have been sorted, id needs to be reassigned from thread_map[id] to 
// pref[group_map[threadToGroup[thread_map[id]]]] - group_size[group_map[threadToGroup[thread_map[id]]]] + localIndexInGroup[new_id]
// do I actually need to sort group_size too??

__global__ void preprocess5(){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if( id >= numbadWarps* 32) return;

    int new_id = thread_map[id];
    int old_grp = threadToGroup[new_id];
    int new_group = group_map[old_grp];
    int grpPos = pref[new_group] - group_size[new_group]; // if I don't sort the sizes as well I can simply use the old group Id stored by threads. BONUS
    
    thread_map[id] = grpPos + localIndexInGroup[new_id];

}

// OPT KERNELS END


__global__ void initialize(int m, int source, int* depth) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(id == source) depth[source] = 0;
		else depth[id] = MYINFINITY;
	}
}

__global__ void bfs_kernel(int m, unsigned long int *row_offsets, int *column_indices, DistT *dist, bool *changed, int depth, int* tmap) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src >= m) return;

    src = tmap[src];

	if(dist[src] == depth - 1) { // visited but not expanded
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];

		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if (dist[dst] > depth) {
				dist[dst] = depth;
				*changed = true;
			}
		}
	}
}


// __global__ void bfs_update(int m, DistT *dist, bool *visited) {
// 	int id = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (id < m) {
// 		if(dist[id] < MYINFINITY && !visited[id])
// 			visited[id] = true;
// 	}
// }

void BFSSolver(Graph &g, int source, DistT* h_dist, int magic_val) {
	//print_device_info(0);
	auto m = g.V();
	auto nnz = g.E();
	auto h_row_offsets = g.out_rowptr();
	auto h_column_indices = g.out_colidx();
	int zero = 0;
	bool one = 1;
	unsigned long *d_row_offsets;
	VertexId *d_column_indices;

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
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

	// Timer t;
	// t.Start();

    float elapsed = 0;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    // Adding OPT Code

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
    preprocess1<<<ceil(((float)totalWarps * 32) / 512), 512>>>(totalWarps * 32, d_P, d_isBad, d_row_offsets, d_numBadWarps, magic_val);
    CUDA_SAFE_CALL(cudaMemcpy(&numBadWarps, d_numBadWarps, sizeof(int), cudaMemcpyDeviceToHost));
    cudaMalloc(&d_Good, numBadWarps * sizeof(int));
    cudaMalloc(&d_Bad, numBadWarps * sizeof(int));
    cudaMemset(d_Gi, 0, sizeof(int));
    cudaMemset(d_Bi, 0, sizeof(int));
    preprocess2<<<ceil(totalWarps * 1.0 / 512), 512>>>(d_isBad, numBadWarps, d_Good, d_Bad, totalWarps, d_Gi, d_Bi);
    CUDA_SAFE_CALL(cudaMemcpy(&Gi, d_Gi, sizeof(int), cudaMemcpyDeviceToHost));
    preprocess2_1<<<max(1, (int)ceil(Gi * 1.0 / 512)), 512>>>(d_P, Gi, d_Good, d_Bad);


    preprocess3<<<nblocks, nthreads>>>(m, numBadWarps, d_row_offsets, thread_mappings, d_P, totalWarps);


    // getting mappings
    int* hmap;
    int numBadBlocks = (numBadWarps*32 - 1)/512 + 1;
    int rmpd = totalWarps*32;
    hmap = (int*)malloc(sizeof(int) * rmpd);
    cudaMemcpy(hmap, thread_mappings, rmpd , cudaMemcpyDeviceToHost);
    for(int i=0;i<numBadBlocks*512; i++){
        std::cout<<hmap[i]<<" -> "<<i<<" "<<h_row_offsets[hmap[i]+1] - h_row_offsets[hmap[i]]<<"\n";
    }
    
    std::cout<<std::endl;
	do {
		++iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy(d_num_frontier, &zero, sizeof(int), cudaMemcpyHostToDevice));   

		bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_dist, d_changed, iter, thread_mappings);
		
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		
	} while (h_changed);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	//t.Stop();
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
 
    cudaEventElapsedTime(&elapsed, start, stop);

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, elapsed);

	CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, m * sizeof(DistT), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
    return;
}