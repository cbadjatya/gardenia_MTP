#!/bin/bash
#PBS -e errorfile_bc.err
#PBS -o logfile_bc.log
#PBS -l select=1:ncpus=32:ngpus=2
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .


for count in {1..5}; do
    source=$(($RANDOM % 100000))
    for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*.mtx; do
   
        for n in {1..5}; do 
            filename=$(basename "$file")
            
            filename_no_ext="${filename%.*}"
            full_file_no_ext="${file%.*}"

            #/lfs/sware/cuda-11.2/bin/nvprof --metrics warp_execution_efficiency,gld_transactions_per_request,gst_transactions_per_request,global_hit_rate,local_hit_rate,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,achieved_occupancy,eligible_warps_per_cycle --log-file bfs_swell_prof.txt ./bin/topo_swell mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_swell_nop.txt" 

            #/lfs/sware/cuda-11.2/bin/nvprof --metrics warp_execution_efficiency,gld_transactions_per_request,gst_transactions_per_request,global_hit_rate,local_hit_rate,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,achieved_occupancy,eligible_warps_per_cycle --log-file bfs_base_prof.txt ./bin/topo_base mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_base_nop.txt" 

            #/lfs/sware/cuda-11.2/bin/nvprof --metrics warp_execution_efficiency,gld_transactions_per_request,gst_transactions_per_request,global_hit_rate,local_hit_rate,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,achieved_occupancy,eligible_warps_per_cycle --log-file bfs_bitonic_prof.txt ./bin/topo_bitonic mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_bitonic_nop.txt" 

            ./bin/bc_topo_swell mtx "$full_file_no_ext" "$source">> "${filename_no_ext}_bc_swell_${source}.txt" 

            ./bin/bc_topo_base mtx "$full_file_no_ext" "$source">> "${filename_no_ext}_bc_base_${source}.txt" 

            ./bin/bc_topo_bitonic mtx "$full_file_no_ext" "$source">> "${filename_no_ext}_bc_bitonic_${source}.txt" 

            ./bin/bc_topo_sort mtx "$full_file_no_ext" "$source">> "${filename_no_ext}_bc_sort_${source}.txt" 

            ./bin/bc_topo_gsort mtx "$full_file_no_ext" "$source">> "${filename_no_ext}_bc_gsort_${source}.txt" 


        
        done
    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
