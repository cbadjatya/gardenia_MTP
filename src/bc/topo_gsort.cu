// Copyright (c) 2016, Xuhao Chen
#include "bc.h"
#include "timer.h"
#include "worklistc.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <vector>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#define BC_VARIANT "topo_gsort"



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


__global__ void initialize(int m, int source, ScoreT *scores, int *path_counts, int *depths, ScoreT *deltas, bool *visited, bool *expanded) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		scores[id] = 0;
		deltas[id] = 0;
		expanded[id] = false;
		if(id == source) {
			visited[id] = true;
			path_counts[id] = 1;
			depths[id] = 0;
		} else {
			visited[id] = false;
			path_counts[id] = 0;
			depths[id] = -1;
		}
	}
}

// Shortest path calculation by forward BFS
__global__ void bc_forward(int m, const int *row_offsets, 
                           const IndexT *column_indices, 
                           int *path_counts, int *depths, int depth, 
                           bool *changed, bool *visited, bool *expanded, 
                           int *nitems, int *queue, int queue_len, int* tmap) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(src < m) src = tmap[src];
	if(src < m && visited[src] && !expanded[src]) {
		expanded[src] = true;
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth)==-1)) {
				int pos = atomicAdd(nitems, 1);
				queue[queue_len + pos] = dst;
				*changed = true;
			}
			if (depths[dst] == depth) {
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

// Dependency accumulation by back propagation
// not sure if optimization should be applied here.
__global__ void bc_reverse(int num, const int *row_offsets, 
                           const IndexT *column_indices, 
                           int start, int *frontiers, 
                           ScoreT *scores, int *path_counts, 
                           int *depths, int depth, ScoreT *deltas) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    
	if(id < num) {
		int src = frontiers[start + id];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		ScoreT delta_src = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if(depths[dst] == depth + 1) {
				delta_src += static_cast<ScoreT>(path_counts[src]) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		deltas[src] = delta_src;
		scores[src] += deltas[src];
	}
}

__global__ void bc_update(int m, int *depths, bool *visited) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
        // id = tmap[id];
		if(depths[id] != -1 && !visited[id])
			visited[id] = true;
	}
}

__global__ void bc_normalize(int m, ScoreT *scores, ScoreT max_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m) scores[tid] = scores[tid] / (max_score);
}

void BCSolver(Graph &g, int source, ScoreT *h_scores) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.out_rowptr();
  auto h_column_indices = g.out_colidx();	
	//print_device_info(0);
	int zero = 0;
	int *d_row_offsets;
  VertexId *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));
	
	ScoreT *d_scores, *d_deltas;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(ScoreT) * m));
	int *d_path_counts, *d_depths, *d_frontiers;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontiers, sizeof(int) * (m+1)));
	bool *d_changed, h_changed, *d_visited, *d_expanded;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_visited, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_expanded, m * sizeof(bool)));
	int *d_nitems, h_nitems = 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_nitems, sizeof(int)));

	int depth = 0;
	int frontiers_len = 0;
	vector<int> depth_index;
	depth_index.push_back(0);
	int nthreads = 512;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, source, d_scores, d_path_counts, d_depths, d_deltas, d_visited, d_expanded);
	// CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_frontiers[0], &source, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA BC solver (%d CTAs/SM, %d threads/CTA) ...\n", nblocks, nthreads);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	
// opt portion

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

	do {
		depth++;
		h_changed = false;
		//printf("iteration=%d, frontire_size=%d\n", depth, h_nitems);
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_nitems, &zero, sizeof(int), cudaMemcpyHostToDevice));
		frontiers_len += h_nitems;
		depth_index.push_back(frontiers_len);
		bc_forward<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_path_counts, d_depths, depth, d_changed, d_visited, d_expanded, d_nitems, d_frontiers, 
        frontiers_len, thread_mappings);
		// CudaTest("solving bc_forward failed");
		bc_update <<<nblocks, nthreads>>> (m, d_depths, d_visited);
		// CudaTest("solving bc_update failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&h_nitems, d_nitems, sizeof(int), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	//printf("\nDone Forward BFS, starting back propagation (dependency accumulation)\n");
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		h_nitems = depth_index[d+1] - depth_index[d];
		//thrust::sort(thrust::device, d_frontiers+depth_index[d], d_frontiers+depth_index[d+1]);
		nblocks = (h_nitems - 1) / nthreads + 1;
		//printf("Reverse: depth=%d, frontier_size=%d\n", d, h_nitems);
		bc_reverse<<<nblocks, nthreads>>>(h_nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
		// CudaTest("solving bc_reverse failed");
	}
	
	//CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	//printf("\nStart calculating the maximum score\n");
	ScoreT *d_max_score;
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
	
	nthreads = 512;
	nblocks = (m - 1) / nthreads + 1;
	bc_normalize<<<nblocks, nthreads>>>(m, d_scores, h_max_score);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());


	printf("\titerations = %d.\n", depth);

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Runtime = %f\n\n\n", milliseconds);

	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_path_counts));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_frontiers));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
}

