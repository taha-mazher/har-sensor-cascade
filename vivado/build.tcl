# Synthesise and implement mlp_relu6 for the paper's part at 50 MHz, non-project batch mode.
# Usage: vivado -mode batch -source build.tcl -tclargs <split>
set split [expr {[llength $argv] ? [lindex $argv 0] : "A"}]
cd [file dirname [file normalize [info script]]]
file mkdir out_$split
# $readmemh default file names resolve from the run directory
foreach f {w1 b1 w2 b2} { file copy -force ../rtl/mem/$split/$f.mem $f.mem }

set part xc7a35ticsg324-1L
read_verilog ../rtl/mlp_relu6.v
read_xdc clock.xdc
synth_design -top mlp_relu6 -part $part
report_utilization -file out_$split/utilization_synth.rpt
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force out_$split/routed.dcp
report_utilization          -file out_$split/utilization_routed.rpt
report_timing_summary       -file out_$split/timing_routed.rpt
report_control_sets -verbose -file out_$split/control_sets.rpt
report_power                -file out_$split/power_vectorless.rpt
write_verilog -mode funcsim -force out_$split/netlist_funcsim.v
