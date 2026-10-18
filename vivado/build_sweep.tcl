# Build one stage-1 variant on split A. Usage: vivado -mode batch -source build_sweep.tcl -tclargs <module> <tag>
# The netlist top is renamed to mlp_relu6 so tb_mlp_relu6.v (NETLIST) simulates every variant unchanged.
lassign $argv module tag
cd [file dirname [file normalize [info script]]]
file mkdir out_$tag
foreach f {w1 b1 w2 b2} { file copy -force ../rtl/mem/A/$f.mem $f.mem }
read_verilog ../rtl/$module.v
read_xdc clock.xdc
synth_design -top $module -part xc7a35ticsg324-1L
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force out_$tag/routed.dcp
report_utilization    -file out_$tag/utilization_routed.rpt
report_timing_summary -file out_$tag/timing_routed.rpt
write_verilog -mode funcsim -rename_top mlp_relu6 -force out_$tag/netlist_funcsim.v
