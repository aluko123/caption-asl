#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=notebook
module load opencv/4.6.0_python3


#SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

#PYTHON_SCRIPT="$SCRIPT_DIR/WLASL/start_kit/preprocess.py"


#python video_downloader.py
#python preprocess.py
#python find_missing.py
python video-gloss-mapping.py
#python "$PYTHON_SCRIPT"



