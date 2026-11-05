# Run from the local directory prepared by python/run_rtl.py.
set_param general.maxThreads 2
create_project -in_memory -part xc7a100tcsg324-1
set_property include_dirs [list [pwd]] [current_fileset]
read_verilog [list hexsight_mac.v hexsight_relu.v hexsight_argmax.v hexsight_cnn.v]
read_xdc core_ooc.xdc
synth_design -top hexsight_cnn -part xc7a100tcsg324-1 -mode out_of_context
report_utilization -file synthesis_utilization.rpt
# Supply a realizable clock-source site for out-of-context timing estimation.
set clock_site [lindex [get_sites -filter {SITE_TYPE == BUFGCTRL}] 0]
if {$clock_site eq ""} { error "No global clock buffer site for this part" }
set_property HD.CLK_SRC $clock_site [get_ports clk]
opt_design
place_design
route_design
report_utilization -file routed_utilization.rpt
report_timing_summary -delay_type min_max -max_paths 10 -report_unconstrained -file routed_timing.rpt
report_drc -file routed_drc.rpt
check_timing -verbose -file check_timing.rpt
# Unlocated OOC ports are not physically routed. Gate internal paths separately;
# retain the full timing report and its provisional port failures for integration.
set sequential [all_registers -clock core_clk]
report_timing -from $sequential -to $sequential -delay_type min_max -max_paths 10 -file internal_timing.rpt
set setup_path [get_timing_paths -from $sequential -to $sequential -delay_type max -max_paths 1]
set hold_path [get_timing_paths -from $sequential -to $sequential -delay_type min -max_paths 1]
set all_setup_wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set all_hold_wns [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {[llength $setup_path] == 0 || [llength $hold_path] == 0} { error "No setup/hold timing paths" }
set setup_wns [get_property SLACK $setup_path]
set hold_wns [get_property SLACK $hold_path]
set dsp [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
set bram36 [llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]
set bram18 [llength [get_cells -hierarchical -filter {REF_NAME == RAMB18E1}]]
set luts [llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]]
set registers [llength [get_cells -hierarchical -filter {REF_NAME =~ FD*}]]
set report [open synthesis_results.json w]
puts $report [format {{"part":"xc7a100tcsg324-1","mode":"out_of_context_routed","clock_period_ns":10.0,"clock_uncertainty_ns":0.1,"input_output_max_delay_ns":2.0,"input_output_min_delay_ns":0.5,"internal_setup_slack_ns":%.6f,"internal_hold_slack_ns":%.6f,"all_paths_setup_slack_ns":%.6f,"all_paths_hold_slack_ns":%.6f,"boundary_timing_closed":false,"dsp48e1":%d,"ramb36e1":%d,"ramb18e1":%d,"lut_primitives":%d,"flip_flops":%d}} $setup_wns $hold_wns $all_setup_wns $all_hold_wns $dsp $bram36 $bram18 $luts $registers]
close $report
if {$setup_wns < 0 || $hold_wns < 0} { error "Internal core timing constraints not met" }
set critical_drc [get_drc_violations -filter {SEVERITY == Error || SEVERITY =~ Critical*}]
if {[llength $critical_drc] != 0} { error "Unresolved critical DRC violations: $critical_drc" }
puts "CORE_INTERNAL_TIMING_PASS"
exit
