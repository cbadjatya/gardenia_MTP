// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#define VC_VARIANT "topo_opt"
#include "vc.h"
#include "timer.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"



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
        if (tid % 32 == 0 && max_value - min_value > magic_val)
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

   __shared__ unsigned long ind[512][2];

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



__global__ void first_fit(int m, int *row_offsets, int *column_indices, int *colors, bool *changed, int* tmap) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;	

	// add mapping here
	if(id >= m) return;
	id = tmap[id];
	bool forbiddenColors[MAXCOLOR+1];
	if (colors[id] == MAXCOLOR) {
		for (int i = 0; i < MAXCOLOR; i++)
			forbiddenColors[i] = false;
		int row_begin = row_offsets[id];
		int row_end = row_offsets[id + 1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = column_indices[offset];
			int color = colors[neighbor];
			forbiddenColors[color] = true;
		}
		int vertex_color;
		for (vertex_color = 0; vertex_color < MAXCOLOR; vertex_color++) {
			if (!forbiddenColors[vertex_color]) {
				colors[id] = vertex_color;
				break;
			}
		}
		assert(vertex_color < MAXCOLOR);
		*changed = true;
	}
}

__global__ void conflict_resolve(int m, int *row_offsets, int *column_indices, int *colors, bool *colored, int* tmap) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// add remapping here
	if(id < m) id = tmap[id];
	if (id < m && !colored[id]) {
		int row_begin = row_offsets[id];
		int row_end = row_offsets[id + 1];
		colored[id] = true;
		for (int offset = row_begin; offset < row_end; offset ++) {
			int neighbor = column_indices[offset];
			if (id < neighbor && colors[id] == colors[neighbor]) {
				colors[id] = MAXCOLOR;
				colored[id] = false;
				break;
			}
		}
	}
}

int VCSolver(Graph &g, int *colors, int magic_val) {
	auto m = g.V();
	auto nnz = g.E();
	auto h_row_offsets = g.out_rowptr();
	auto h_column_indices = g.out_colidx();	
	//print_device_info(0);
	int *d_row_offsets, *d_column_indices, *d_colors;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, m * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_colors, colors, m * sizeof(int), cudaMemcpyHostToDevice));
	bool *d_changed, h_changed, *d_colored;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colored, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_colored, 0, m * sizeof(bool)));

	int num_colors = 0, iter = 0;
	int nthreads = 512;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA VC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	// Timer t;
	// t.Start();	

	cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

	// Adding OPT Code

    int *P, *d_P, numBadWarps, *d_numBadWarps;
    bool *isBad, *d_isBad;
    int totalWarps = (m / 32);
    P = (int *)malloc(totalWarps * sizeof(int));
    isBad = (bool *)malloc(totalWarps);

    numBadWarps = 0;
    cudaMalloc(&d_numBadWarps, sizeof(int));
    cudaMemset(d_numBadWarps, 0, sizeof(int));

    cudaMalloc(&d_P, totalWarps * sizeof(int));

    cudaMalloc(&d_isBad, totalWarps);
    cudaMemset(d_isBad, 0, totalWarps);

    preprocess1<<<ceil(((float)totalWarps * 32) / 512), 512>>>(totalWarps * 32, d_P, d_isBad, d_row_offsets, d_numBadWarps, magic_val);
    // CHECK_DEBUG(cudaMemcpy(P,d_P,totalWarps*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&numBadWarps, d_numBadWarps, sizeof(int), cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(isBad,d_isBad,totalWarps,cudaMemcpyDeviceToHost));

    int *d_Good, *d_Bad, *d_Gi, *d_Bi;
    int Gi, Bi;

    cudaMalloc(&d_Good, numBadWarps * sizeof(int));
    cudaMalloc(&d_Bad, numBadWarps * sizeof(int));
    cudaMalloc(&d_Gi, sizeof(int));
    cudaMalloc(&d_Bi, sizeof(int));
    cudaMemset(d_Gi, 0, sizeof(int));
    cudaMemset(d_Bi, 0, sizeof(int));

    preprocess2<<<ceil(totalWarps * 1.0 / 512), 512>>>(d_isBad, numBadWarps, d_Good, d_Bad, totalWarps, d_Gi, d_Bi);

    CUDA_SAFE_CALL(cudaMemcpy(&Gi, d_Gi, sizeof(int), cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(&Bi, d_Bi,sizeof(int), cudaMemcpyDeviceToHost));

    // cudaFree(d_isBad);

    preprocess2_1<<<max(1, (int)ceil(Gi * 1.0 / 512)), 512>>>(d_P, Gi, d_Good, d_Bad);

    // cudaFree(d_Good);
    // cudaFree(d_Bad);

    // CHECK(cudaMemcpy(P,d_P,totalWarps*sizeof(int),cudaMemcpyDeviceToHost));
    int* tmap;
    cudaMalloc(&tmap, m*sizeof(int));
    preprocess3<<<nblocks, nthreads>>>(m, numBadWarps, d_row_offsets, tmap, d_P, totalWarps);

    // OPT code ends
    // int* h_tm;
    // h_tm = (int*)malloc(sizeof(int)*m);
    // cudaMemcpy(h_tm, thread_mappings, sizeof(int)*m, cudaMemcpyDeviceToHost);

    // sort(h_tm, h_tm+(int)(ceil(numBadWarps*32.0/512)*512), comp(h_row_offsets));

    // cudaMemcpy(thread_mappings, h_tm, sizeof(int)*m, cudaMemcpyHostToDevice);


    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // float elapsed = 0;
    // cudaEventElapsedTime(&elapsed, start, stop);

    // printf("time taken for opt = %f\n",elapsed);
    // printf("Number of Bad Warps found = %d\n", numBadWarps);


	do {
		iter ++;
		//printf("iteration=%d\n", iter);
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		first_fit<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_colors, d_changed, tmap);
		// CudaTest("first_fit failed");
		conflict_resolve<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_colors, d_colored, tmap);
		// CudaTest("conflict_resolve failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	// t.Stop();

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);


	CUDA_SAFE_CALL(cudaMemcpy(colors, d_colors, m * sizeof(int), cudaMemcpyDeviceToHost));
	#pragma omp parallel for reduction(max : num_colors)
	for (int n = 0; n < m; n ++)
		num_colors = max(num_colors, colors[n]);
	num_colors ++;	
	printf("\titerations = %d.\n", iter);
	printf("\truntime[%s] = %f ms, num_colors = %d.\n", VC_VARIANT, elapsed, num_colors);
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_colors));
	return num_colors;
}
