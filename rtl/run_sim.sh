#!/bin/bash
# Bit-exact check of mlp_relu6.v on all three test sets. Run from WSL inside hw_fix/rtl.
# Uses a system iverilog if installed, else the unpacked copy in ~/iverilog-local.
set -e
cd "$(dirname "$0")"
if command -v iverilog >/dev/null; then IV=iverilog; VVP=vvp; IVB=; VVM=
else
    R=~/iverilog-local/root; L=$R/usr/lib/x86_64-linux-gnu
    export LD_LIBRARY_PATH=$L
    IV=$R/usr/bin/iverilog; VVP=$R/usr/bin/vvp; IVB=-B$L/ivl; VVM=-M$L/ivl
fi
for s in A B C; do
    $IV $IVB -g2005 -Wall -P tb_mlp_relu6.MEM_DIR=\"mem/$s/\" -P tb_mlp_relu6.TEST_FILE=\"../out/test_$s.mem\" \
        -o /tmp/sim_$s tb_mlp_relu6.v mlp_relu6.v
    $VVP $VVM /tmp/sim_$s | grep -E "samples=|RESULT|MISMATCH"
done
