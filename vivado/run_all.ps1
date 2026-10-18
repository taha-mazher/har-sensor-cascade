# Build, gate-level bit-exact simulation with activity capture, and activity-based power for one split.
# Usage (PowerShell, from anywhere): .\run_all.ps1 A
param([string]$Split = "A")
$ErrorActionPreference = "Stop"
$bin = "C:\Xilinx\Vivado\2024.2\bin"
Set-Location $PSScriptRoot
$out = "out_$Split"

& "$bin\vivado.bat" -mode batch -nojournal -log "build_$Split.log" -source build.tcl -tclargs $Split | Out-Null

# netlist sim runs from hw_fix\vivado so the testbench's ../out/test_<split>.mem path resolves
& "$bin\xvlog.bat" -d NETLIST ../rtl/tb_mlp_relu6.v "$out/netlist_funcsim.v" "C:/Xilinx/Vivado/2024.2/data/verilog/src/glbl.v" | Out-Null
& "$bin\xelab.bat" -debug typical -L unisims_ver -L secureip tb_mlp_relu6 glbl -s "tb_net_$Split" | Out-Null
$env:SPLIT_OUT = $out
# options go through a file: cmd.exe splits arguments on '=' inside the .bat wrapper
Set-Content -Encoding ascii "xsim_opts_$Split.txt" "-testplusarg TEST=../out/test_$Split.mem -tclbatch saif.tcl -log xsim_$Split.log"
& "$bin\xsim.bat" "tb_net_$Split" -f "xsim_opts_$Split.txt" | Out-Null
Select-String -Path "xsim_$Split.log" -Pattern "samples=|RESULT" | ForEach-Object { $_.Line }

& "$bin\vivado.bat" -mode batch -nojournal -log "power_$Split.log" -source power.tcl -tclargs $Split | Out-Null
