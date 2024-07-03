#!/bin/bash
#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .


for n in {1..5}; do 
    for file in /lfs1/usrscratch/mtech/cs22m036/snap/*csr*.txt; do
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"

        ./swell "$file"  >> "${filename_no_ext}_swell.txt" 
        ./base "$file"  >> "${filename_no_ext}_base.txt" 

        continue;

        ./bitonic "$file"  >> "${filename_no_ext}_bitonic.txt" 
        ./gsort "$file"  >> "${filename_no_ext}_gsort.txt"

        ./nop_base "$file"  >> "${filename_no_ext}_nop_base.txt" 
        ./nop_bitonic "$file"  >> "${filename_no_ext}_nop_bitonic.txt" 
        ./nop_gsort "$file"  >> "${filename_no_ext}_nop_gsort.txt"
        
    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
