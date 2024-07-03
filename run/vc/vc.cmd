#!/bin/bash
#PBS -e errorfile_vc.err
#PBS -o logfile_vc.log
#PBS -l select=1:ncpus=64:ngpus=2
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .



for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*.mtx; do
    for n in {1..5}; do 
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"
    
        ./bin/vc_topo_gsort mtx "$full_file_no_ext" 700 >> "${filename_no_ext}_vc_gsort.txt" 
        ./bin/vc_topo_sort mtx "$full_file_no_ext" >> "${filename_no_ext}_vc_sort.txt" 



       
    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
