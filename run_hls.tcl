open_project -reset lab4_csim
set_top kernel_gemm
add_files mars_wide_bus.h
add_files kernel_gemm.cpp
add_files -tb testbench.cpp


open_solution -reset "0_naive"
open_solution -reset "1_tiling"
open_solution -reset "2_pipelining"
open_solution -reset "3_unrolling"
open_solution -reset "4_inlining"
open_solution -reset "5_partitioning"

set_part {xcu50-fsvh2104-2-e}

create_clock -period 3.33

csim_design
csynth_design
#cosim_design

exit
