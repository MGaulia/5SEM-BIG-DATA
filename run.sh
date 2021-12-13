#!/bin/sh
#SBATCH -p short
#SBATCH --ntasks=8
#SBATCH --output=res.out
#SBATCH --error=res.err

mpirun python3 app.py