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

__global__ void populate(int N, int* row_offsets, int* column_indices, int* new_ind, int* tmap, int* nnz, int* warp_base, int* check, int* rev_map){

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N) return;


    int nid = tmap[id]; // nid is the vertex this thread is supposed to process
    int wid = id / 32;
    int wsz = min(32, N - wid*32);

    int wbase = warp_base[wid] + id % wsz; // the base address for this vertex
   
    int row_begin = row_offsets[nid];
    int row_end = row_offsets[nid + 1];

    for(int i = row_begin; i<row_end; i++){
        int ind = rev_map[column_indices[i]];
        new_ind[wbase] = ind;
        atomicAdd(&check[wbase],1);
        
        wbase += wsz;
    }

    // this mapping might be called ELL-32-C (or some such thing where 32 is sigma)

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

// nnz has neighbors stored in the sorted manner. access using id not nid.
__global__ void broadcast_graph_swell(int p, int N, int *phase, int * new_column_indices, int *parent, int *discovered, int* tmap, int* warp_base, int* nnz)
{

    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id >= N) return;

    // int nid = tmap[id];

    // if (phase[nid] != p) return;

    int wid = id/32;
    int wsz = min(32, N - wid*32);
    int wbase = warp_base[wid] + id%wsz;

    int num_neighbors = nnz[id];

    
    for (int j =  0; j < num_neighbors; j++)
    {
        int nbr = new_column_indices[wbase];
        if (phase[nbr] < 0 && phase[id]==p)
        {
            phase[nbr] = p + 1;
            // parent[nbr] = nid;
            *discovered = 0;
        }
        wbase += wsz;
        // __syncwarp();
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

    // sort nodes in increasing order of number of neighbors

    int* nnz;
    int* tmap;

    nnz = (int*)malloc(N * sizeof(int)); // number of non zeroes (no. of neighbors for each node)
    
    tmap = (int*)malloc(N * sizeof(int)); // remapping id for each vertex

    for(int i=0;i<N;i++){
        nnz[i] = row_offsets[i+1] - row_offsets[i];
        tmap[i] = i;
    }
    int* rev_map = (int*)malloc(N * sizeof(int));

    thrust::sort_by_key(nnz, nnz + N, tmap, thrust::greater<int>()); // nnz and tmap are both sorted


    for(int i=0;i<N;i++){
        rev_map[tmap[i]] = i;
    }


    int* warp_sizes;
    int num_warps = (N-1)/32 + 1;
    warp_sizes = (int*)malloc(sizeof(int) * num_warps);
    int tot_size = 0;

    for(int i=0;i<N;i+=32){
        warp_sizes[i/32] = nnz[i]*32;
        tot_size += nnz[i]*32;
    }
    // the last warp won't be fully filled
    if(N%32 != 0){
        int x = (N/32) * 32;
        
         warp_sizes[N/32 + 1] = nnz[x]*(N-x);
         tot_size += nnz[x]*(N-x);
    }
    
    int* warp_base = (int*)malloc(sizeof(int) * num_warps);
    thrust::exclusive_scan(warp_sizes, warp_sizes + num_warps, warp_base);

    int* d_row_offsets, *d_column_indices, *d_new_ind, *d_tmap, *d_nnz, *d_warp_base, *d_rev_map;
    cudaMalloc(&d_row_offsets, (N+1) * sizeof(int));
    cudaMalloc(&d_column_indices, F_sz * sizeof(int));
    cudaMalloc(&d_new_ind, tot_size * sizeof(int));
    cudaMalloc(&d_tmap, N * sizeof(int));
    cudaMalloc(&d_rev_map, N * sizeof(int));
    cudaMalloc(&d_nnz, N * sizeof(int));
    cudaMalloc(&d_warp_base, num_warps * sizeof(int));

    cudaMemcpy(d_row_offsets, row_offsets, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, column_indices, F_sz * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_ind, new_ind, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmap, tmap, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rev_map, rev_map, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnz, nnz, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_warp_base, warp_base, num_warps * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (N-1)/32 + 1;
    int num_threads = 32;

    // implementing a memory check...was any index in d_new_ind accessed more/less than once
    int* check = (int*)malloc(sizeof(int) * tot_size);
    int* d_check;
    cudaMalloc(&d_check, sizeof(int) * tot_size);
    cudaMemset(d_check, 0, sizeof(int) * tot_size);


    populate<<<num_blocks,32>>>(N, d_row_offsets, d_column_indices, d_new_ind, d_tmap, d_nnz, d_warp_base, d_check, d_rev_map); // add d_check if necessary
    CHECK_DEBUG(cudaMemcpy(check,d_check, tot_size * sizeof(int), cudaMemcpyDeviceToHost));
    bool check_failed = false;
    for(int i=0; i<tot_size; i++){
        if(check[i] > 1){
            check_failed = true;
            cout<<i<<" "<<check[i]<<"\n";
        }
    }
    if(check_failed == true) return 0;

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
    
        initialize_graph<<<num_blocks, num_threads>>>(N, parent_d, phase_d, rev_map[ROOT]);

        cudaError_t err = cudaGetLastError(); // Get error code

        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();

        discovered = 0;
        float milliseconds = 0;

        p = 0;

        cudaEventRecord(start);

        while (!discovered)
        {

            discovered = 1;
            cudaMemcpy(discovered_d, &discovered, sizeof(int), cudaMemcpyHostToDevice);

            broadcast_graph_swell<<<num_blocks, num_threads>>>(p, N, phase_d, d_new_ind, parent_d, discovered_d, d_tmap, d_warp_base, d_nnz);

            CHECK(cudaMemcpy(&discovered, discovered_d, sizeof(int), cudaMemcpyDeviceToHost));
            p++;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Swell Runtime = %f\n", milliseconds);
        printf("------------------------Total no. of phases : %d ------------------------\n\n\n", p);
        // printf("Avg time taken by the kernels : %f ms\n\n", avg_time / iter);
        cudaMemcpy(phase_swell, phase_d, N * sizeof(int), cudaMemcpyDeviceToHost);

        return 0;
}
