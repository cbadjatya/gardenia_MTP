// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
// Topology-driven Minimum Spanning Tree using CUDA
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
#include "common.h"
#include "graph_io.h"
#include "gbar.h"
#include "component.h"
#include "cuda_launch_config.hpp"

__global__ void dinit(int m, foru *eleminwts, foru *minwtcomponent, unsigned *partners, bool *processinnextiteration, unsigned *goaheadnodeofcomponent) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		eleminwts[id] = MYINFINITY;
		minwtcomponent[id] = MYINFINITY;
		partners[id] = id;
		goaheadnodeofcomponent[id] = m;
		processinnextiteration[id] = false;
	}
}

__global__ void dfindelemin(int m, int *row_offsets, int *column_indices, foru *weight, unsigned *mstwt, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		unsigned src = id;
		unsigned srcboss = cs.find(src);
		unsigned dstboss = m;
		foru minwt = MYINFINITY;
		unsigned row_begin = row_offsets[src];
		unsigned row_end = row_offsets[src + 1];
		for (unsigned offset = row_begin; offset < row_end; ++ offset) {
			foru wt = weight[offset];
			if (wt < minwt) {
				unsigned dst = column_indices[offset];
				unsigned tempdstboss = cs.find(dst);
				if (srcboss != tempdstboss) {
					minwt = wt;
					dstboss = tempdstboss;
				}
			}
		}
		eleminwts[id] = minwt;
		partners[id] = dstboss;
		if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {
			atomicMin(&minwtcomponent[srcboss], minwt);
		}
	}
}

__global__ void dfindelemin2(int m, int *row_offsets, int *column_indices, foru *weight, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *goaheadnodeofcomponent) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		unsigned src = id;
		unsigned srcboss = cs.find(src);
		if(eleminwts[id] == minwtcomponent[srcboss] && srcboss != partners[id] && partners[id] != m) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1];
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				foru wt = weight[offset];
				if (wt == eleminwts[id]) {
					unsigned dst = column_indices[offset];
					unsigned tempdstboss = cs.find(dst);
					if (tempdstboss == partners[id]) {
						atomicCAS(&goaheadnodeofcomponent[srcboss], m, id);
					}
				}
			}
		}
	}
}

__global__ void verify_min_elem(int m, int *row_offsets, int *column_indices, foru *weight, unsigned *mstwt, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, bool *processinnextiteration, unsigned *goaheadnodeofcomponent) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(cs.isBoss(id)) {
			if(goaheadnodeofcomponent[id] == m) {
				return;
			}
			unsigned minwt_node = goaheadnodeofcomponent[id];
			foru minwt = minwtcomponent[id];
			if(minwt == MYINFINITY)
				return;
			bool minwt_found = false;
			unsigned row_begin = row_offsets[minwt_node];
			unsigned row_end = row_offsets[minwt_node + 1];
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				foru wt = weight[offset];
				if (wt == minwt) {
					minwt_found = true;
					unsigned dst = column_indices[offset];
					unsigned tempdstboss = cs.find(dst);
					if(tempdstboss == partners[minwt_node] && tempdstboss != id) {
						processinnextiteration[minwt_node] = true;
						return;
					}
				}
			}
		}
	}
}

/*
__global__ void elim_dups(int m, int *row_offsets, int *column_indices, foru *weight, unsigned *mstwt, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < m) id = inpid;
	if (id < m) {
		if(processinnextiteration[id]) {
			unsigned srcc = cs.find(id);
			unsigned dstc = partners[id];
			if(minwtcomponent[dstc] == eleminwts[id]) {
				if(id < goaheadnodeofcomponent[dstc]) {
					processinnextiteration[id] = false;
				}
			}
		}
	}
}

__global__ void dfindcompmin(int m, int *row_offsets, int *column_indices, foru *weight, unsigned *mstwt, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, bool *processinnextiteration, unsigned *goaheadnodeofcomponent) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		if(partners[id] == m)
			return;
		unsigned srcboss = cs.find(id);
		unsigned dstboss = cs.find(partners[id]);
		if (id != partners[id] && srcboss != dstboss && eleminwts[id] != MYINFINITY && minwtcomponent[srcboss] == eleminwts[id] && dstboss != id && goaheadnodeofcomponent[srcboss] == id) {
			if(!processinnextiteration[id]);
		}
		else {
			if(processinnextiteration[id]);
		}
	}
}
*/

__global__ void dfindcompmintwo(int m, int *row_offsets, int *column_indices, foru *weight, unsigned *mstwt, ComponentSpace csw, foru *eleminwts, foru *minwtcomponent, unsigned *partners, bool *processinnextiteration, GlobalBarrier gb, bool *repeat, unsigned *count) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id, nthreads = blockDim.x * gridDim.x;
	unsigned up = (m + nthreads - 1) / nthreads * nthreads;
	unsigned srcboss, dstboss;
	for(id = tid; id < up; id += nthreads) {
		if(id < m && processinnextiteration[id]) {
			srcboss = csw.find(id);
			dstboss = csw.find(partners[id]);
		}
		gb.Sync();
		/*
		if (id < m && processinnextiteration[id] && srcboss != dstboss) {
			//printf("trying unify id=%d (%d -> %d)\n", id, srcboss, dstboss);
			if (csw.unify(srcboss, dstboss)) {
				atomicAdd(mstwt, eleminwts[id]);
				atomicAdd(count, 1);
				//printf("u %d -> %d (%d)\n", srcboss, dstboss, eleminwts[id]);
				processinnextiteration[id] = false;
				eleminwts[id] = MYINFINITY;	// mark end of processing to avoid getting repeated.
			}
			else {
				*repeat = true;
			}
			//printf("\tcomp[%d] = %d.\n", srcboss, csw.find(srcboss));
		}
		gb.Sync();
		*/
	}
}

int main(int argc, char *argv[]) {
	printf("Minimum Spanning Tree (MST) with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> <device(0/1)>\n", argv[0]);
		exit(1);
	}
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL;
	foru *h_weight = NULL;
	if (strstr(argv[1], ".mtx"))
		mtx2csr(argv[1], m, nnz, h_row_offsets, h_column_indices, h_weight);
	else if (strstr(argv[1], ".graph"))
		graph2csr(argv[1], m, nnz, h_row_offsets, h_column_indices, h_weight);
	else if (strstr(argv[1], ".gr"))
		gr2csr(argv[1], m, nnz, h_row_offsets, h_column_indices, h_weight);
	else { printf("Unrecognizable input file format\n"); exit(0); }

	int device = 0;
	if (argc > 2) device = atoi(argv[2]);
	assert(device == 0 || device == 1); 
	int deviceCount = 0;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
	CUDA_SAFE_CALL(cudaSetDevice(device));
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	int nSM = deviceProp.multiProcessorCount;
	fprintf(stdout, "Found %d devices, using device %d (%s), compute capability %d.%d, cores %d*%d.\n", 
			deviceCount, device, deviceProp.name, deviceProp.major, deviceProp.minor, nSM, ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
	foru *d_weight;
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, nnz * sizeof(foru)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, nnz * sizeof(foru), cudaMemcpyHostToDevice));
	
	unsigned *mstwt, hmstwt = 0;
	int iteration = 0;
	unsigned *partners;
	foru *eleminwts, *minwtcomponent;
	bool *processinnextiteration;
	unsigned *goaheadnodeofcomponent;
	double starttime, endtime;
	ComponentSpace cs(m);
	unsigned prevncomponents, currncomponents = m;
	bool repeat = false, *grepeat;
	unsigned edgecount = 0, *gedgecount;

	CUDA_SAFE_CALL(cudaMalloc((void **)&mstwt, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpy(mstwt, &hmstwt, sizeof(hmstwt), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&eleminwts, m * sizeof(foru)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&minwtcomponent, m * sizeof(foru)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&partners, m * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&processinnextiteration, m * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&goaheadnodeofcomponent, m * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc(&grepeat, sizeof(bool) * 1));
	CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc(&gedgecount, sizeof(unsigned) * 1));
	CUDA_SAFE_CALL(cudaMemcpy(gedgecount, &edgecount, sizeof(unsigned) * 1, cudaMemcpyHostToDevice));

	int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	//const size_t max_blocks = maximum_residency(dfindcompmintwo, nthreads, 0);
	const size_t max_blocks = 4;
	printf("Setup global barrier, nSM=%d, max_blocks=%d\n", nSM, max_blocks);
	GlobalBarrierLifetime gb;
	gb.Setup(nSM * max_blocks);
	printf("Finding mst...\n");
	starttime = rtclock();
	do {
		++iteration;
		prevncomponents = currncomponents;
		dinit<<<nblocks, nthreads>>>(m, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent);
		CudaTest("dinit failed");
		dfindelemin<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_weight, mstwt, cs, eleminwts, minwtcomponent, partners);
		dfindelemin2<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_weight, cs, eleminwts, minwtcomponent, partners, goaheadnodeofcomponent);
		verify_min_elem<<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_weight, mstwt, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent);
		CudaTest("dfindelemin failed");
		do {
			repeat = false;
			CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));
			dfindcompmintwo <<<nSM * max_blocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_weight, mstwt, cs, eleminwts, minwtcomponent, partners, processinnextiteration, gb, grepeat, gedgecount);
			CudaTest("dfindcompmintwo failed");
			CUDA_SAFE_CALL(cudaMemcpy(&repeat, grepeat, sizeof(bool) * 1, cudaMemcpyDeviceToHost));
		} while (repeat); // only required for quicker convergence?
		currncomponents = cs.numberOfComponentsHost();
		CUDA_SAFE_CALL(cudaMemcpy(&hmstwt, mstwt, sizeof(hmstwt), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&edgecount, gedgecount, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost));
		printf("\titeration %d, number of components = %d (%d), mstwt = %u mstedges = %u\n", iteration, currncomponents, prevncomponents, hmstwt, edgecount);
	} while (currncomponents != prevncomponents);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();

	printf("\tmstwt = %u, iterations = %d.\n", hmstwt, iteration);
	printf("\t%s result: weight: %u, components: %u, edges: %u\n", argv[1], hmstwt, currncomponents, edgecount);
	printf("\truntime [mst] = %f ms.\n", 1000 * (endtime - starttime));
	return 0;
}
