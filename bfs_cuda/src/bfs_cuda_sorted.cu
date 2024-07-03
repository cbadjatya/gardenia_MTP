#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <thrust/sort.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>
// any definitions left?

using namespace std;


// using global definitions for better sorting
int* CSR_N, *phase, p;

__global__ void initialize_refs(int N, int* CSR_N, int* key, int* val){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N){
		key[i] = CSR_N[i+1] - CSR_N[i];
		val[i] = i;
	}
}

__global__ void initialize_graph(int N, int* parent, int* phase, int ROOT){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<N){
		parent[i] = -1;
		phase[i] = -1;
		if(i==ROOT){
			phase[ROOT] = 0;
		}
	}
}


__global__ void broadcast_graph(int p, int N, int* phase, int* CSR_N, int* CSR_F, int* parent, int* discovered, int* mapping_d){
	
                        
                                
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int i = tid;
	if(i < N){
		
		i = mapping_d[i];
        if(phase[i]==p)
		for(int j = 0; j < (CSR_N[i+1]-CSR_N[i]); j++){
			int nbr = CSR_F[j+CSR_N[i]];
			if(phase[nbr] < 0){
				phase[nbr] = p+1;
				parent[nbr] = i;
				*discovered = 0;
			
			}
		}
		
	}

}

// int comp(const void* i, const void* j){
//         const int* a = (const int*)i;
//         const int* b = (const int*)j;
// 	int x = (int)(CSR_N[*a+1]-CSR_N[*a]);
// 	int y = (int)(CSR_N[*b+1]-CSR_N[*b]);
// 	return x - y;
// }

void printSolution(int* phase ,int* parent, int N){
	for(int i=0;i<N;i++){
		printf("node %d -> parent = %d, phase = %d\n",i,parent[i],phase[i]);
	}
}

void readGraph(char* filename, int* N, int* ROOT, int** CSR_N, int** CSR_F, int* N_sz, int* F_sz){ //verified

	  FILE* in = fopen(filename, "r");
    fscanf(in, "%d\n", N);
    fscanf(in, "%d\n", N_sz);
    fscanf(in, "%d\n", F_sz);
    fscanf(in, "%d\n", ROOT);
    *CSR_N = (int*) malloc(*N_sz*sizeof(int));
    *CSR_F = (int*) malloc(*F_sz*sizeof(int));

    for(int i=0;i<*N_sz;i++) fscanf(in, "%d ", &((*CSR_N)[i]));
    for(int i=0;i<*F_sz;i++) fscanf(in, "%d ", &((*CSR_F)[i]));

    fclose(in);

}

int main(int argc, char* argv[]){

	int N,ROOT;
	int N_sz, F_sz;
        int* CSR_F;
	cudaEvent_t start, stop;


	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	readGraph(argv[1], &N, &ROOT, &CSR_N, &CSR_F, &N_sz, &F_sz);
	int* parent, *parent_d, *phase_d;
	int* CSR_N_d, *CSR_F_d;
	parent = (int*)malloc(N*sizeof(int));
	phase = (int*)malloc(N*sizeof(int));

	cudaMalloc((void**)&parent_d, N*sizeof(int));
	cudaMalloc((void**)&phase_d, N*sizeof(int));
	cudaMalloc((void**)&CSR_N_d, N_sz*sizeof(int));
	cudaMalloc((void**)&CSR_F_d, F_sz*sizeof(int));


	cudaMemcpy(CSR_N_d, CSR_N, N_sz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(CSR_F_d, CSR_F, F_sz*sizeof(int), cudaMemcpyHostToDevice);

	p = 0;

	int discovered = 1;
	int* discovered_d;

	cudaMalloc((void**)&discovered_d, sizeof(int));
	cudaMemcpy(discovered_d, &discovered, sizeof(int), cudaMemcpyHostToDevice);

	int num_threads = 512;
	int num_blocks = ceil((N*1.0)/num_threads);

	initialize_graph<<< num_blocks, num_threads >>> (N, parent_d, phase_d, ROOT);
	//cudaError_t err = cudaGetLastError();        // Get error code

	//if ( err != cudaSuccess )
	//{
	//	printf("CUDA Error: %s\n", cudaGetErrorString(err));
	//	exit(-1);
	//}
	cudaDeviceSynchronize();
  
  // creating a mapping for reference redirection

	discovered = 0;
	float milliseconds = 0;

	// clock_t t;
	// t = clock();
	cudaEventRecord(start);

	int* mapping_d;  
	cudaMalloc(&mapping_d, N*sizeof(int));
	
	int* keys;
	cudaMalloc(&keys, N*sizeof(int));
	
	initialize_refs<<<num_blocks, num_threads>>>(N, CSR_N_d, keys, mapping_d);
	cudaDeviceSynchronize();

	thrust::sort_by_key(thrust::device, keys, keys + N, mapping_d);

	while(!discovered){
    
		discovered = 1;
		cudaMemcpy(discovered_d, &discovered, sizeof(int), cudaMemcpyHostToDevice);
		broadcast_graph <<<num_blocks, num_threads>>> (p, N, phase_d, CSR_N_d, CSR_F_d, parent_d, discovered_d, mapping_d);
		//cudaDeviceSynchronize();
		cudaMemcpy(&discovered, discovered_d, sizeof(int), cudaMemcpyDeviceToHost);
		p++;
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f\n", milliseconds);

	// printf("Total no. of phases : %d\n", iter);
	// printf("Avg time taken by the kernels : %f ms\n\n", avg_time / iter);
	cudaMemcpy(phase, phase_d, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(parent, parent_d, N * sizeof(int), cudaMemcpyDeviceToHost);
	fprintf(stderr,"%d %d %d\n",phase, parent[ROOT], phase[ROOT]);

	return 0;


}
