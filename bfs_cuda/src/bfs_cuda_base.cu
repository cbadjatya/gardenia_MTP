#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <thrust/sort.h>
#include <thrust/scan.h>


#define NUM_TESTS 1

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


void readGraph(char *filename, int *N, int *ROOT, int **row_offsets, int **column_indices, int *N_sz, int *F_sz)
{ // verified

    FILE *in = fopen(filename, "r");
    fscanf(in, "%d\n", N);
    fscanf(in, "%d\n", N_sz);
    fscanf(in, "%d\n", F_sz);
    fscanf(in, "%d\n", ROOT);
    *row_offsets = (int *)malloc(*N_sz * sizeof(int));
    *column_indices = (int *)malloc(*F_sz * sizeof(int));

    for (int i = 0; i < *N_sz; i++)
        fscanf(in, "%d ", &((*row_offsets)[i]));
    for (int i = 0; i < *F_sz; i++)
        fscanf(in, "%d ", &((*column_indices)[i]));

    fclose(in);
}



__global__ void initialize_graph(int N, int *parent, int *phase, int ROOT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        parent[i] = -1;
        phase[i] = -1;
        if (i == ROOT)
        {
            phase[ROOT] = 0;
        }
    }
}


__global__ void broadcast_graph_base(int p, int N, int *phase, int *row_offsets, int *column_indices, int *parent, int *discovered)
{

    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id >= N) return;

    // if (phase[id] != p) return;

    int start = row_offsets[id];
    int end = row_offsets[id+1];

    for (int j =  start; j < end; j++)
    {
        int nbr = column_indices[j];
        if (phase[nbr] < 0 && phase[id] == p)
        {
            phase[nbr] = p + 1;
            // parent[nbr] = id;
            *discovered = 0;
        }
    }
}

int main(int argc, char *argv[])
{
    int N, ROOT;
    int N_sz, F_sz;
    int *column_indices;
    int *phase_base, *row_offsets, p, *phase_swell;

    readGraph(argv[1], &N, &ROOT, &row_offsets, &column_indices, &N_sz, &F_sz);
    
    int *parent, *parent_d, *phase_d;
    // parent = (int *)malloc(N * sizeof(int));
    phase_swell = (int *)malloc(N * sizeof(int));
    phase_base = (int *)malloc(N * sizeof(int));


    int* d_row_offsets, *d_column_indices;
    cudaMalloc(&d_row_offsets, (N+1) * sizeof(int));
    cudaMalloc(&d_column_indices, F_sz * sizeof(int));
   

    cudaMemcpy(d_row_offsets, row_offsets, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, column_indices, F_sz * sizeof(int), cudaMemcpyHostToDevice);


    int num_blocks = (N-1)/32 + 1;
    int num_threads = 32;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&parent_d, N * sizeof(int));
    cudaMalloc((void **)&phase_d, N * sizeof(int));

    p = 0;

        int discovered = 1;
        int *discovered_d;

        cudaMalloc((void **)&discovered_d, sizeof(int));
        cudaMemcpy(discovered_d, &discovered, sizeof(int), cudaMemcpyHostToDevice);

        // int num_threads = 32;
        // int num_blocks = ceil((N * 1.0) / num_threads);

        initialize_graph<<<num_blocks, num_threads>>>(N, parent_d, phase_d, ROOT);
        cudaError_t err = cudaGetLastError(); // Get error code

        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();

        discovered = 0;
        float milliseconds = 0;

        cudaEventRecord(start);

        while (!discovered)
        {

            discovered = 1;
            cudaMemcpy(discovered_d, &discovered, sizeof(int), cudaMemcpyHostToDevice);

            broadcast_graph_base<<<num_blocks, num_threads>>>(p, N, phase_d, d_row_offsets, d_column_indices, parent_d, discovered_d);

            CHECK(cudaMemcpy(&discovered, discovered_d, sizeof(int), cudaMemcpyDeviceToHost));
            p++;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Base Runtime = %f\n", milliseconds);
        printf("------------------------Total no. of phases : %d ------------------------\n\n\n", p);
        // printf("Avg time taken by the kernels : %f ms\n\n", avg_time / iter);
        cudaMemcpy(phase_base, phase_d, N * sizeof(int), cudaMemcpyDeviceToHost);

        return 0;

}
