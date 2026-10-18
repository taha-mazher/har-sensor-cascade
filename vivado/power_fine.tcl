# Same as power.tcl but in mW with more digits, for comparing small dynamic powers. Usage: -tclargs <tag> <tb>
lassign $argv tag tb
cd [file dirname [file normalize [info script]]]
open_checkpoint out_$tag/routed.dcp
read_saif out_$tag/activity.saif -strip_path $tb/dut
set_units -power mW
report_power -file out_$tag/power_saif_mw.rpt
