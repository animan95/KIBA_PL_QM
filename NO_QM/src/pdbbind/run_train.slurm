#!/bin/bash
#SBATCH --job-name=PROTML
#SBATCH --time=12:29:00
#SBATCH --nodes=1 --ntasks=95
#SBATCH --account=PAS0291
#SBATCH --output=%x.%j

module load python

cd $SLURM_SUBMIT_DIR
cd /users/PAS0291/aniketmandal95/prot-lig/src/pdbbind
#rm -rf build
#./configure release intel mkl openmp nolibintok nointracule nomgc nonewpath2 noccman2

pip install torch torch-geometric rdkit pandas scikit-learn optuna tensorboard

#python pred_race_test.py
python trainer.py > rmse.txt
