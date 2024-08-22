#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=notebook
#module load opencv/4.6.0_python3
#module load python/3.8
#module load scipy-stack
#module load sklearn/1.8.0

source /apps/profiles/modules_asax.sh.dyn
module load anaconda/3-2024.02
module load pytorch

#SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

#PYTHON_SCRIPT="$SCRIPT_DIR/WLASL/start_kit/preprocess.py"


#python video_downloader.py
#python preprocess.py
#python find_missing.py
#python semantic_analysis.py
python input_nlp.py
#python transform_encoder.py
#python english_asl.py
#python parallel_dataset.py
#python "$PYTHON_SCRIPT"



