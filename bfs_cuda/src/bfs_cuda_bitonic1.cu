#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define NUM_TESTS 1

using namespace std;


// using global definitions for better sorting
int* CSR_N, *phase, p;

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


__global__ void broadcast_graph(int p, int N, int* phase, int* CSR_N, int* CSR_F, int* parent, int* discovered){


    // instead of just one warp playing, how about there are at least 4 to tango?                           
     // Remapping / Balancing Code ---- apply only if this is a full block. If N - blockIdx.x * blockDim.x < 512, go forth with a normal implemementation.
     // let the flag be int and be updated by atomics, turn the opt on if it is at least 4 (or some other heuristic)
    unsigned int i, ij, id, size;
	__shared__ int ind[512][2];
	__shared__ int flag; 

    i = threadIdx.x;
	id = i + blockIdx.x * blockDim.x;
	
	if(i == 0) flag = 0;
	size = 512;
	if(N - blockIdx.x * blockDim.x >= 512){
		
   		 
		ind[threadIdx.x][0] = (phase[id]==p)*(CSR_N[id+1]-CSR_N[id]); //the loop's limit
    	ind[threadIdx.x][1] = id;
		int max_value = ind[i][0];
		int min_value = ind[i][0];
		for (int i=16; i>0; i=i/2){
			max_value = max(max_value, __shfl_down_sync(-1, max_value, i));
			min_value = min(min_value, __shfl_down_sync(-1, min_value, i));
		}
		int warp_score = max_value - min_value;
		if(warp_score > 700 && threadIdx.x % 32 == 0) atomicAdd(&flag,1); // if any one of the warp threads meet the condition do the deed.
	}
   
   __syncthreads();
   if(flag >= 2)
    for(int k = 2; k<= size; k<<=1){
      for(int j = k>>1; j > 0; j = j >> 1){

           ij = i ^ j;

           if (ij > i)
           {   
               int temp[2];
               if (((i & k) == 0 && ind[i][0] > ind[ij][0]) ||((i & k) != 0 && ind[i][0] < ind[ij][0]))
               {
                       temp[0] = ind[i][0]; temp[1] = ind[i][1];
                       ind[i][0] = ind[ij][0]; ind[i][1] = ind[ij][1];
                       ind[ij][0] = temp[0]; ind[ij][1] = temp[1];
               }
           }
           __syncthreads();
       }
    }
   
    if(N - blockIdx.x * blockDim.x >= 512){
		id = ind[threadIdx.x][1];
	}
	
	if(id < N){
		int lim = (phase[id]==p)*(CSR_N[id+1]-CSR_N[id]);
		for(int j = 0; j < lim; j++){
			int nbr = CSR_F[j+CSR_N[id]];
			if(phase[nbr] < 0){
				phase[nbr] = p+1;
				parent[nbr] = id;
				*discovered = 0;
			
			}
		}
//		for(int j = 0; j < (phase[id]==p)*(CSR_N[id+1]-CSR_N[id])*100; j++){
//			int x = 20;
//			x = x + 10;
//			// useless operations.
//		}
		
	}

}


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
    int* CSR_N, *phase, p;

	readGraph(argv[1], &N, &ROOT, &CSR_N, &CSR_F, &N_sz, &F_sz);
	int* parent, *parent_d, *phase_d;
	int* CSR_N_d, *CSR_F_d;
	parent = (int*)malloc(N*sizeof(int));
	phase = (int*)malloc(N*sizeof(int));

    for(int i=0;i<NUM_TESTS;i++)
    {
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

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
        cudaError_t err = cudaGetLastError();        // Get error code

        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();


        discovered = 0;
        int iter = 0;
        float avg_time = 0;
        float milliseconds = 0;

        cudaEventRecord(start);
        while(!discovered){

            discovered = 1;
            cudaMemcpy(discovered_d, &discovered, sizeof(int), cudaMemcpyHostToDevice);

            broadcast_graph <<<num_blocks, num_threads>>> (p, N, phase_d, CSR_N_d, CSR_F_d, parent_d, discovered_d);
            //cudaDeviceSynchronize();
           // if ( err != cudaSuccess )
           // {
            //    printf("CUDA Error: %s\n", cudaGetErrorString(err));
             //   exit(-1);
           // }		
            cudaMemcpy(&discovered, discovered_d, sizeof(int), cudaMemcpyDeviceToHost);
            
            p++;


            cudaMemcpy(phase, phase_d, N*sizeof(int), cudaMemcpyDeviceToHost);

            //printf("%d ",p);
        }
        //printf("\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        //t = clock() - t;
        //double time_taken = ((double)t)/CLOCKS_PER_SEC;
        printf("%f\n",milliseconds);

        //printf("Total no. of phases : %d\n",iter);  
        //printf("Avg time taken by the kernels : %f ms\n\n",avg_time/iter);
        cudaMemcpy(phase, phase_d, N*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, parent_d, N*sizeof(int), cudaMemcpyDeviceToHost);
        fprintf(stderr,"%d %d %d\n",phase, parent[ROOT], phase[ROOT]);
        //printSolution(phase, parent, N);
    }
	return 0;
}
