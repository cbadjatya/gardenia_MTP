#!/bin/bash
#PBS -e errorfile_sssp.err
#PBS -o logfile_sssp.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q nvk_gpuq
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .


for n in {1..1}; do 
    for file in /lfs1/usrscratch/mtech/cs22m036/weighted_mtx/*.mtx; do
        filename=$(basename "$file")
        filename_no_ext="${filename%.*}"
        full_file_no_ext="${file%.*}"
        ./bin/sssp_topo_base mtx "$full_file_no_ext" >> "${filename_no_ext}_sssp_topo_base.txt" 
        

        ./bin/sssp_topo_opt mtx "$full_file_no_ext" 700 >> "${filename_no_ext}_sssp_topo_opt.txt" 

        continue

        ./bin/sssp_topo_opt mtx "$full_file_no_ext" 700 >> "${filename_no_ext}_sssp_topo_opt.txt" 
        ./bin/sssp_topo_opt mtx "$full_file_no_ext" 1000 >> "${filename_no_ext}_sssp_topo_opt.txt" 

        ./bin/sssp_topo_opt mtx "$full_file_no_ext" 1500 >> "${filename_no_ext}_sssp_topo_opt.txt" 
        ./bin/sssp_topo_opt mtx "$full_file_no_ext" 3000 >> "${filename_no_ext}_sssp_topo_opt.txt" 
        ./bin/sssp_topo_opt mtx "$full_file_no_ext" 3500 >> "${filename_no_ext}_sssp_topo_opt.txt" 
        ./bin/sssp_topo_opt mtx "$full_file_no_ext" 4000 >> "${filename_no_ext}_sssp_topo_opt.txt" 
        ./bin/sssp_topo_opt mtx "$full_file_no_ext" 5000 >> "${filename_no_ext}_sssp_topo_opt.txt" 

       
    done
done

mv -f * $PBS_O_WORKDIR/.
rm -rf $tempdir 
