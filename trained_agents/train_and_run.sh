#!/bin/bash
#SBATCH --time=2:00:00

# sbatch --array=1-${n} train_and_run.sh ${output_dir} ${filename} ${train} ${run}

spack load py-torch py-cloudpickle@2 py-h5py
source ../../lunarlander/bin/activate

output_dir=$1
filename=$2
train=$3
run=$4
i=${SLURM_ARRAY_TASK_ID}

cd ..

if [ "$train" = true ] ; then
  echo $(date +"%Y-%m-%d %T") " Training agent ${filename}_${i}"
  # Run the command with the current suffix in the filename
  python train_agent.py --f "${output_dir}/${filename}_${i}" $flags_train
fi
if [ "$run" = true ] ; then
  echo $(date +"%Y-%m-%d %T") " Running agent ${filename}_${i}"
  # Run the command with the current suffix in the filename
  python run_agent.py --f "${output_dir}/${filename}_${i}" $flags_run
fi
echo $(date +"%Y-%m-%d %T") " Agent ${filename}_${i} completed."
