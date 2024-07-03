#!/bin/bash
#PBS -e errorfile_bfs.err
#PBS -o logfile_bfs.log
#PBS -l select=1:ncpus=64:ngpus=2
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .

#lscpu > cpuinfo.txt

for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*wiki*.mtx; do
   
    for n in {1..1}; do 
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"

        /lfs/sware/cuda-11.2/bin/nvprof --metrics warp_execution_efficiency,gld_transactions_per_request,gst_transactions_per_request,global_hit_rate,local_hit_rate,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,achieved_occupancy,eligible_warps_per_cycle --log-file bfs_swell_prof.txt ./bin/topo_swell mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_swell.txt" 

        /lfs/sware/cuda-11.2/bin/nvprof --metrics warp_execution_efficiency,gld_transactions_per_request,gst_transactions_per_request,global_hit_rate,local_hit_rate,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,achieved_occupancy,eligible_warps_per_cycle --log-file bfs_base_prof.txt ./bin/topo_base mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_base.txt" 

        #/lfs/sware/cuda-11.2/bin/nvprof --metrics warp_execution_efficiency,gld_transactions_per_request,gst_transactions_per_request,global_hit_rate,local_hit_rate,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,achieved_occupancy,eligible_warps_per_cycle --log-file bfs_bitonic_prof.txt ./bin/topo_bitonic mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_bitonic_nop.txt" 

        #./bin/topo_swell mtx "$full_file_no_ext"  >> "${filename_no_ext}_bfs_swell.txt" 

        #./bin/topo_base mtx "$full_file_no_ext"  >> "${filename_no_ext}_bfs_base.txt" 

        
        #./bin/topo_bitonic mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_bitonic.txt" 

        #./bin/topo_sort mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_sort.txt" 

        #./bin/topo_gsort mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_gsort.txt" 


    
    done
done



mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
