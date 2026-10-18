#!/bin/bash
# Bit-exact check of the three stage-1 variants (serial, neuron-parallel, fully parallel) on all splits. Run from WSL.
set -e
cd "$(dirname "$0")"
R=~/iverilog-local/root; L=$R/usr/lib/x86_64-linux-gnu; export LD_LIBRARY_PATH=$L
IV="$R/usr/bin/iverilog -B$L/ivl"; VVP="$R/usr/bin/vvp -M$L/ivl"
for m in mlp_relu6 mlp_relu6_np mlp_relu6_par; do
  for s in A B C; do
    $IV -g2005 -DDUT=$m -P tb_mlp_relu6.MEM_DIR=\"mem/$s/\" -P tb_mlp_relu6.TEST_FILE=\"../out/test_$s.mem\" -o /tmp/sw_$m$s tb_mlp_relu6.v $m.v
    echo "$m $s: $($VVP /tmp/sw_$m$s | grep -E 'RESULT|latency' | tr '\n' ' ' | sed 's/.*mismatches/mismatches/')"
  done
done
