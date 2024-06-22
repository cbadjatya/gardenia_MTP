#!/bin/bash
#PBS -e errorfile_vc.err
#PBS -o logfile_vc.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .



#for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*kitter*.mtx; do
for file in datasets/chesapeake.mtx; do 
    for n in {1..1}; do 
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"
        ./bin/scc_base "$file" >> "${filename_no_ext}_scc_base.txt" 
        #./bin/vc_opt "$full_file_no_ext" 700 >> "${filename_no_ext}_vc_opt.txt" 

       
    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
