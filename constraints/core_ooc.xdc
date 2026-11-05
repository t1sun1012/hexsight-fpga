# Core-only timing assumptions, not Nexys A7 board pin constraints.
# Port delays are provisional. Unlocated OOC boundary paths are not signoff paths;
# scripts/synth_core.tcl checks internal paths and preserves all boundary reports.
create_clock -name core_clk -period 10.000 [get_ports clk]
set_clock_uncertainty 0.100 [get_clocks core_clk]
set_input_delay -clock core_clk -max 2.000 [get_ports {rst start pixel_valid pixel_data[*]}]
set_input_delay -clock core_clk -min 0.500 [get_ports {rst start pixel_valid pixel_data[*]}]
set_output_delay -clock core_clk -max 2.000 [all_outputs]
set_output_delay -clock core_clk -min 0.500 [all_outputs]
