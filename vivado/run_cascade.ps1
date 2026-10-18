# Build, gate-level bit-exact simulation with activity capture, and activity-based power for one cascade model.
# Usage (PowerShell): .\run_cascade.ps1 cascade_uci
param([string]$Name = "cascade_uci")
$ErrorActionPreference = "Stop"
$bin = "C:\Xilinx\Vivado\2024.2\bin"
Set-Location $PSScriptRoot
$out = "out_$Name"

& "$bin\vivado.bat" -mode batch -nojournal -log "build_$Name.log" -source build_cascade.tcl -tclargs $Name | Out-Null

# the netlist testbench gets N2 as a plusarg; plusargs go through a file (cmd.exe splits on '=')
$n2 = (Get-Content "../rtl/mem/$Name/params.txt") -replace '^N2=(\d+).*', '$1'
& "$bin\xvlog.bat" -d NETLIST ../rtl/tb_cascade.v "$out/netlist_funcsim.v" "C:/Xilinx/Vivado/2024.2/data/verilog/src/glbl.v" | Out-Null
& "$bin\xelab.bat" -debug typical -L unisims_ver -L secureip tb_cascade glbl -s "tb_net_$Name" | Out-Null
$env:SPLIT_OUT = $out
Set-Content -Encoding ascii "xsim_opts_$Name.txt" "-testplusarg TEST=../out/$Name.mem -testplusarg N2=$n2 -tclbatch saif_cascade.tcl -log xsim_$Name.log"
& "$bin\xsim.bat" "tb_net_$Name" -f "xsim_opts_$Name.txt" | Out-Null
Select-String -Path "xsim_$Name.log" -Pattern "windows=|RESULT" | ForEach-Object { $_.Line }

& "$bin\vivado.bat" -mode batch -nojournal -log "power_$Name.log" -source power_cascade.tcl -tclargs $Name | Out-Null
