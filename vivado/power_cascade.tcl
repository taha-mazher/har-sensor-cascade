# Power of the routed design using the simulated switching activity
set split [expr {[llength $argv] ? [lindex $argv 0] : "A"}]
cd [file dirname [file normalize [info script]]]
open_checkpoint out_$split/routed.dcp
read_saif out_$split/activity.saif -strip_path tb_cascade/dut
report_power -file out_$split/power_saif.rpt
