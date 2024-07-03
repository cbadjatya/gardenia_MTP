#!/bin/bash
#PBS -e errorfile_sgd.err
#PBS -o logfile_sgd.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .


for n in {1..1}; do 
    for file in /lfs1/usrscratch/mtech/cs22m036/weighted_mtx/*web*.mtx; do
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"
        ./bin/sgd_base "$file" >> "${filename_no_ext}_sgd_base.txt" 
        #./bin/sgd_opt "$file" >> "${filename_no_ext}_sgd_opt.txt" 

    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
