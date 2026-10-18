# Synthesise and implement cascade.v for one exported model at 50 MHz.
# Usage: vivado -mode batch -source build_cascade.tcl -tclargs <name>   (name = cascade_uci, cascade_pamap2_104, ...)
set name [lindex $argv 0]
cd [file dirname [file normalize [info script]]]
file mkdir out_$name
foreach f {w b} { file copy -force ../rtl/mem/$name/$f.mem $f.mem }
set fp [open ../rtl/mem/$name/params.txt]; set gen [string trim [read $fp]]; close $fp

read_verilog ../rtl/cascade.v
read_xdc clock.xdc
synth_design -top cascade -part xc7a35ticsg324-1L -generic $gen
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force out_$name/routed.dcp
report_utilization          -file out_$name/utilization_routed.rpt
report_utilization -hierarchical -file out_$name/utilization_hier.rpt
report_timing_summary       -file out_$name/timing_routed.rpt
write_verilog -mode funcsim -force out_$name/netlist_funcsim.v
