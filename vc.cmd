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



for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*.mtx; do
    for n in {1..1}; do 
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"
        ./bin/vc_topo_base mtx "$full_file_no_ext" >> "${filename_no_ext}_vc_topo_base.txt" 
        ./bin/vc_opt mtx "$full_file_no_ext" 700 >> "${filename_no_ext}_vc_opt.txt" 

       
    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
