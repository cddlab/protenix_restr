#!/bin/bash
#SBATCH -J pnx_ex
#SBATCH -o run_restr_example.out
#SBATCH -e run_restr_example.err
#SBATCH -p q1
#SBATCH --gres=gpu:1
# protenix RGI example runner. GPU work must go through sbatch (not the login node).
# Submit from THIS repo directory:  cd protenix_restr && sbatch run_restr_example.sh
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source .venv/bin/activate

rm -rf out_restr_example
# restr_example.json nests `restraints_config` and runs single-sequence (--use_msa false).
# RGI: rgi_utils minimizes distance + conformer restraints on the x0 prediction each step.
protenix pred -i restr_example.json -o out_restr_example \
    --use_default_params true --use_msa false --seeds 0 --step 200 --sample 1 --cycle 4 \
    2>&1 | grep -iE "rgi_utils|built spec|setup:|finalize|Error|Traceback" || true

CIF=$(find out_restr_example -name '*.cif' | head -1)
echo "prediction: $CIF"
echo done
