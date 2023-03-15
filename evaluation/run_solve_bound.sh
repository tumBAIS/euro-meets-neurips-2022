#!/bin/bash
#SBATCH -J ml-co
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --mail-type=end
#SBATCH --mem=40000mb
#SBATCH --cpus-per-task=24
#SBATCH --export=NONE
#SBATCH --time=70:00:00

module load python/3.8.11-base
module load gcc/11.2.0
module load slurm_setup
source ../venv/bin/activate

export CUDA_VISIBLE_DEVICES=""

for INSTANCE_SEED in ${INSTANCE_SEEDS}
do
	for ITERATION in ${TRAINING_ITERATIONS}
	do
		MODEL_NAME="${STRATEGY}_samples-${SAMPLESIZE}_instances-${NUMINSTANCES}_runtime-${RUNTIMESTRATEGY}_featureset-${FEATURE_SET}_iteration-${ITERATION}"
		python solve_bound.py --strategy=$STRATEGY --model_name=$MODEL_NAME --instance_directory=$INSTANCE_DIRECTORY --time_limit=$TIME_LIMIT --result_directory=$RESULT_DIRECTORY --instance_seed=$INSTANCE_SEED --monte_carlo_sampling_rounds=9 --model_directory=$DIR_MODELS
	done
done
