#!/bin/bash
#PBS -e errorfile_sssp.err
#PBS -o logfile_sssp.log
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
       
        ./bin/sssp_topo_gsort mtx "$full_file_no_ext" >> "${filename_no_ext}_sssp_topo_gsort.txt" 
        #./bin/sssp_topo_sort mtx "$full_file_no_ext" >> "${filename_no_ext}_sssp_topo_sort.txt" 

       
    done
done


mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
