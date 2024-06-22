// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "cc.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"


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

   __shared__ int ind[256][2];

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

        for (int k = 2; k <= 256; k <<= 1)
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

__global__ void hook(int m, const int *row_offsets, 
                     const int *column_indices, 
                     int *comp, bool *changed, int* tmap) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
        src = tmap[src];
		int comp_src = comp[src];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			int comp_dst = comp[dst];
			// int comp_dst = __ldg(comp+dst);
			if (comp_src == comp_dst) continue;
			int high_comp = comp_src > comp_dst ? comp_src : comp_dst;
			int low_comp = comp_src + (comp_dst - high_comp);
			if (high_comp == comp[high_comp]) {
				*changed = true;
				comp[high_comp] = low_comp;
			}
		}
	}
}

__global__ void shortcut(int m, int *comp) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < m) {
		while (comp[src] != comp[comp[src]]) {
			comp[src] = comp[comp[src]];
		}
	}
}

void CCSolver(Graph &g, int *h_comp) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.out_rowptr();
  auto h_column_indices = g.out_colidx();	
  //print_device_info(0);
  int *d_row_offsets;
  int *d_column_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
  int *d_comp;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, sizeof(int) * m));
  CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, m * sizeof(int), cudaMemcpyHostToDevice));
  bool h_changed, *d_changed;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));

  int iter = 0;
  int nthreads = 256;
  int nblocks = (m - 1) / nthreads + 1;
  printf("Launching CUDA CC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

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
    preprocess1<<<ceil(((float)totalWarps * 32) / 256), 256>>>(totalWarps * 32, d_P, d_isBad, d_row_offsets, d_numBadWarps, 350);
    CUDA_SAFE_CALL(cudaMemcpy(&numBadWarps, d_numBadWarps, sizeof(int), cudaMemcpyDeviceToHost));
    cudaMalloc(&d_Good, numBadWarps * sizeof(int));
    cudaMalloc(&d_Bad, numBadWarps * sizeof(int));
    cudaMemset(d_Gi, 0, sizeof(int));
    cudaMemset(d_Bi, 0, sizeof(int));
    preprocess2<<<ceil(totalWarps * 1.0 / 256), 256>>>(d_isBad, numBadWarps, d_Good, d_Bad, totalWarps, d_Gi, d_Bi);
    CUDA_SAFE_CALL(cudaMemcpy(&Gi, d_Gi, sizeof(int), cudaMemcpyDeviceToHost));
    preprocess2_1<<<max(1, (int)ceil(Gi * 1.0 / 256)), 256>>>(d_P, Gi, d_Good, d_Bad);


    preprocess3<<<nblocks, nthreads>>>(m, numBadWarps, d_row_offsets, thread_mappings, d_P, totalWarps);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("opt time = %f\n", milliseconds);

    // check if mapping is correct
    // int* hmap;
    // hmap = (int*)malloc(sizeof(int) * m);
    // cudaMemcpy(hmap, thread_mappings, sizeof(int) * m, cudaMemcpyDeviceToHost);

    // int *vis;
    // vis = (int*)malloc(sizeof(int) * m);
    // for(int i=0;i<m;i++) vis[i] = i;

    // for(int i=0;i<m;i++){
    //     cout<<i<<" "<<hmap[i]<<"\n";
    // }


    cudaEventRecord(start);

  do {
    ++ iter;
    h_changed = false;
    CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
    //printf("iteration=%d\n", iter);
    hook<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_comp, d_changed, thread_mappings);
    // CudaTest("solving kernel hook failed");
    shortcut<<<nblocks, nthreads>>>(m, d_comp);
    // CudaTest("solving kernel shortcut failed");
    CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
  } while (h_changed);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed = 0;
  cudaEventElapsedTime(&elapsed, start, stop);
	
	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n\n\n", "bitonic", elapsed);

  // printf("\titerations = %d.\n", iter);
  // printf("\truntime [cuda_base] = %f ms.\n", t.Millisecs());
  CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(int) * m, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_row_offsets));
  CUDA_SAFE_CALL(cudaFree(d_column_indices));
  CUDA_SAFE_CALL(cudaFree(d_changed));
}

