#!/bin/bash
#PBS -e errorfile_spmv.err
#PBS -o logfile_spmv.log
#PBS -l select=1:ncpus=64:ngpus=2
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .


for n in {1..5}; do 
    for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*.mtx; do
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"

        ./bin/spmv_bitonic mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_bitonic.txt" 

        ./bin/spmv_sort mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_sort.txt" 
        ./bin/spmv_gsort mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_gsort.txt" 

    done
done

for n in {1..1}; do 
    for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*rgg*.mtx; do
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"

        /lfs/sware/cuda-11.2/bin/nvprof --metrics warp_execution_efficiency,gld_transactions_per_request,gst_transactions_per_request,global_hit_rate,local_hit_rate,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,achieved_occupancy,eligible_warps_per_cycle --log-file profiler_output_swell.txt ./bin/spmv_swell mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_swell.txt" 
       
        /lfs/sware/cuda-11.2/bin/nvprof --metrics warp_execution_efficiency,gld_transactions_per_request,gst_transactions_per_request,global_hit_rate,local_hit_rate,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,achieved_occupancy,eligible_warps_per_cycle --log-file profiler_output_base.txt ./bin/spmv_base mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_base.txt" 
        

    done
done


mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
