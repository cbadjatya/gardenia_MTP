#!/bin/bash
#PBS -e errorfile_cc.err
#PBS -o logfile_cc.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .



for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*.mtx; do
#for file in datasets/chesapeake.mtx; do
for n in {1..30}; do 
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"

        ./bin/cc_base mtx "$full_file_no_ext" >> "${filename_no_ext}_cc_base.txt" 
        ./bin/cc_bitonic mtx "$full_file_no_ext" >> "${filename_no_ext}_cc_bitonic.txt" 
        ./bin/cc_gsort mtx "$full_file_no_ext" >> "${filename_no_ext}_cc_gsort.txt" 

       
    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
