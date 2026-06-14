#!/bin/bash
#SBATCH -J pnx_grp
#SBATCH -o run_grp_test.out
#SBATCH -e run_grp_test.err
#SBATCH -p q3
#SBATCH --gres=gpu:1
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source .venv/bin/activate
rm -rf out_grp_test
if ! protenix pred -i grp_test.json -o out_grp_test --use_default_params true --use_msa false --seeds 0 --step 200 --sample 1 --cycle 4 > run_grp_test.log 2>&1; then
  echo "protenix FAILED:"; tail -n 40 run_grp_test.log; exit 1
fi
grep -iE "built spec|setup:|finalize" run_grp_test.log || true
CIF=$(find out_grp_test -name '*.cif' | head -1); echo "CIF: $CIF"
GP=../chai-lab_restr/.venv/bin/python
"$GP" ../check_angle.py "$CIF" 5-84 90-180 186-224 || true
"$GP" ../check_dihedral.py "$CIF" 5-50 51-100 101-150 151-224 || true
echo done
