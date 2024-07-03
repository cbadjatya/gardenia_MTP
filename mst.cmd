#!/bin/bash
#PBS -e errorfile1.err
#PBS -o logfile1.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .

for n in {1..1}; do 
    #for file in /lfs1/usrscratch/mtech/cs22m036/mtx/*.mtx; do
    for file in datasets/chesapeake.mtx; do
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"
        ./bin/mst_main "$file" >> "${filename_no_ext}_bc_topo_base.txt" 
        
    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
