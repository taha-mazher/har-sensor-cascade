# xsim batch: record switching activity of the implemented netlist over the whole test set
open_saif $::env(SPLIT_OUT)/activity.saif
log_saif [get_objects -r /tb_cascade/dut/*]
run all
close_saif
quit
