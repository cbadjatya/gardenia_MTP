#!/bin/bash
#PBS -e errorfile_bfs.err
#PBS -o logfile_bfs.log
#PBS -l select=1:ncpus=32:ngpus=2
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .

lscpu > cpuinfo.txt

for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*.mtx; do
for n in {1..10}; do 
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"

        #./bin/topo_base mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_base.txt" 

        #./bin/topo_swell mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_swell.txt" 
        
        #./bin/topo_bitonic mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_bitonic.txt" 

        ./bin/topo_sort mtx "$full_file_no_ext" >> "${filename_no_ext}_bfs_sort.txt" 


       
    done
done



mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
