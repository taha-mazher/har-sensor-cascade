# Parallelism sweep for stage 1 (split A): build, gate-level bit-exact sim with SAIF, activity-based power.
# Usage: .\run_sweep.ps1 mlp_relu6_np A_np
param([string]$Module, [string]$Tag)
$ErrorActionPreference = "Stop"
$bin = "C:\Xilinx\Vivado\2024.2\bin"
Set-Location $PSScriptRoot
$out = "out_$Tag"
& "$bin\vivado.bat" -mode batch -nojournal -log "build_$Tag.log" -source build_sweep.tcl -tclargs $Module $Tag | Out-Null
& "$bin\xvlog.bat" -d NETLIST ../rtl/tb_mlp_relu6.v "$out/netlist_funcsim.v" "C:/Xilinx/Vivado/2024.2/data/verilog/src/glbl.v" | Out-Null
& "$bin\xelab.bat" -debug typical -L unisims_ver -L secureip tb_mlp_relu6 glbl -s "tb_net_$Tag" | Out-Null
$env:SPLIT_OUT = $out
Set-Content -Encoding ascii "xsim_opts_$Tag.txt" "-testplusarg TEST=../out/test_A.mem -tclbatch saif.tcl -log xsim_$Tag.log"
& "$bin\xsim.bat" "tb_net_$Tag" -f "xsim_opts_$Tag.txt" | Out-Null
Select-String -Path "xsim_$Tag.log" -Pattern "samples=|RESULT" | ForEach-Object { $_.Line }
& "$bin\vivado.bat" -mode batch -nojournal -log "power_$Tag.log" -source power.tcl -tclargs $Tag | Out-Null
