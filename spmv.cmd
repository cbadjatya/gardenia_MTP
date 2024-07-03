#!/bin/bash
#PBS -e errorfile_spmv.err
#PBS -o logfile_spmv.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .


for n in {1..1}; do 
    for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*.mtx; do
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"
        #./bin/spmv_base mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_base.txt" 
        ./bin/spmv_opt mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_opt_20.txt" 

    done
done

for n in {1..1}; do 
    for file in /lfs1/usrscratch/mtech/cs22m036/weighted_mtx/*.mtx; do
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"
        ./bin/spmv_base mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_base.txt" 
        ./bin/spmv_opt mtx "$full_file_no_ext" >> "${filename_no_ext}_spmv_opt_700.txt" 

    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
