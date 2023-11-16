#!/bin/bash
#SBATCH --mem=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=21:0:0    
#SBATCH --mail-user=huiyi.wang@mcgill.ca
#SBATCH --mail-type=ALL

source ~/python-environments/myo_env/bin/activate
cd ~/MyoLeg_Sarcopenia

python train_leg_local.py