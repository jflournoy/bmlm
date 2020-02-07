#!/bin/bash
#
#SBATCH --job-name=seastan
#SBATCH -o %x_%A.log
#SBATCH --time=9-00:00:00
#SBATCH --cpus-per-task=7
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=ncf
#SBATCH --mem=20000
#SBATCH --mail-type=END

module load gcc/7.1.0-fasrc01
module load R/3.5.1-fasrc01

export OMP_NUM_THREADS=7

srun Rscript --verbose sea_mlm.R
