#!/bin/bash
#SBATCH -J pnx_rawp
#SBATCH -o run_raw_prot.out
#SBATCH --gres=gpu:1
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source .venv/bin/activate
rm -rf out_raw_prot
protenix pred -i raw_prot.json -o out_raw_prot --use_default_params true --use_msa false --seeds 0 --step 200 --sample 1 --cycle 4 > run_raw_prot.log 2>&1 || { echo FAILED; tail -5 run_raw_prot.log; exit 1; }
CIF=$(find out_raw_prot -name '*.cif' | head -1)
python -c "import gemmi,math; st=gemmi.read_structure('$CIF'); ps=[a.pos for m in st for ch in m for r in ch for a in r]; nan=sum(1 for p in ps if math.isnan(p.x)); print(f'RAW PROTEIN-ONLY: atoms={len(ps)} nan={nan} -> {\"NaN\" if nan else \"CLEAN\"}')"
