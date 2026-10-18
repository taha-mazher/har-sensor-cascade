#!/bin/bash
# Bit-exact check of cascade.v on every exported model (rtl/mem/cascade_*). Run from WSL.
set -e
cd "$(dirname "$0")"
if command -v iverilog >/dev/null; then IV=iverilog; VVP=vvp; IVB=; VVM=
else
    R=~/iverilog-local/root; L=$R/usr/lib/x86_64-linux-gnu
    export LD_LIBRARY_PATH=$L
    IV=$R/usr/bin/iverilog; VVP=$R/usr/bin/vvp; IVB=-B$L/ivl; VVM=-M$L/ivl
fi
for d in mem/cascade_*/; do
    name=$(basename "$d")
    P=$(tr -d '\r\n' < "$d/params.txt" | sed 's/\([A-Z_0-9]*\)=/-P tb_cascade.\1=/g')
    $IV $IVB -g2005 -Wall $P -P tb_cascade.MEM_DIR=\"$d\" -P tb_cascade.TEST_FILE=\"../out/$name.mem\" \
        -o /tmp/sim_$name tb_cascade.v cascade.v
    $VVP $VVM /tmp/sim_$name | grep -E "windows=|RESULT|MISMATCH"
done
