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

python run_training.py --oracle_solutions_directory=${DIR_ORACLE} --dir_models=${DIR_MODELS} --num_perturbations=${NUM_PERTURBATIONS} --sd_perturbation=${SD_PERTURBATION} --time_limit=${TIME_LIMIT} --learning_rate=${LEARNING_RATE} --predictor=${PREDICTOR} --feature_set=${FEATURE_SET}
