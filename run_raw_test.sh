#!/bin/bash
#SBATCH -J pnx_raw
#SBATCH -o run_raw_test.out
#SBATCH -e run_raw_test.err
#SBATCH --gres=gpu:1
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source .venv/bin/activate
DT="${1:-bf16}"; OUT="out_raw_${DT}"; rm -rf "$OUT"
DFLAG=""; [ "$DT" = "fp32" ] && DFLAG="-d fp32"
echo "=== raw protenix (NO restraints), dtype=$DT ==="
if protenix pred -i raw_test.json -o "$OUT" $DFLAG --use_default_params true --use_msa false --seeds 0 --step 200 --sample 1 --cycle 4 > "run_raw_${DT}.log" 2>&1; then
  CIF=$(find "$OUT" -name '*.cif' | head -1)
  python -c "import gemmi,sys,math; st=gemmi.read_structure('$CIF'); xs=[a.pos for m in st for ch in m for r in ch for a in r]; print('atoms:',len(xs),'any_nan:', any(math.isnan(p.x) for p in xs))"
  echo "raw $DT: FOLDED OK"
else
  echo "raw $DT: FAILED"; grep -iE "nan|error|non-finite" "run_raw_${DT}.log" | tail -5
fi
