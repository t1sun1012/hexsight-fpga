# Step 4: implement and verify the Verilog CNN

Completed November 5, 2026. This step turns the [Step 3 integer reference](step-03.md) into a synthesizable hardware datapath. The core computes convolution, activation rescaling, pooling, dense scores, and the predicted class. Python is used for preparation and verification, not for calculations inside the RTL.

The CNN topology, exported parameters, class order, and fixed-point arithmetic are unchanged.

## Architecture

The implementation shares one signed multiply-accumulate unit between convolution and the dense layer:

    Sequential pixel input -> 256-byte pixel memory
                              |
    Convolution weights ------+--> shared signed MAC --> rounded ReLU
                                                          |
                                                 convolution memory
                                                          |
                                                   max-pool control
                                                          |
                                                    pooled memory
                                                          |
    Dense weights ----------------> shared signed MAC --> signed argmax
                                                          |
                                                   prediction + done

A finite-state machine controls the addresses, memory reads, multiplication, accumulation, and stage transitions. The same physical multiplier processes each term over time.

Modules:

| File | Responsibility |
|---|---|
| [hexsight_cnn.v](../rtl/hexsight_cnn.v) | Input loading, memories, convolution/pooling/dense addressing, sequencing, completion |
| [hexsight_mac.v](../rtl/hexsight_mac.v) | Registered signed 16x16 multiplication and signed 40-bit accumulation |
| [hexsight_relu.v](../rtl/hexsight_relu.v) | Rounded shift by 12, ReLU, and saturation to 32767 |
| [hexsight_argmax.v](../rtl/hexsight_argmax.v) | Ordered signed score comparison; lowest class index wins ties |

RTL files use Verilog. The testbenches use SystemVerilog for assertions and test tasks.

This is a fixed architecture, not a configurable CNN generator. Changing only fixed_point.vh does not resize the controller. Weight-file paths are module parameters; the dimensions and numeric contract must remain compatible.

## Clock-by-clock arithmetic

Every MAC term uses three controller states:

1. READ: synchronous memory outputs receive the input feature and weight.
2. MUL: the shared multiplier registers the signed 32-bit product.
3. ADD: the product is sign-extended and added to the 40-bit accumulator.

Before the first term of each convolution output or dense score, the controller loads the correctly scaled, sign-extended bias. It does not add the bias repeatedly.

Pixel bytes are zero-extended to signed 16 bits. This prevents bright pixels from being interpreted as negative numbers. Pooled activations and weights are already signed 16-bit values.

Convolution uses eight 3x3 kernels without flipping, padding, or stride changes. Its 8 x 14 x 14 results are stored in channel, row, column order.

After the last term, rounded ReLU uses an extended 41-bit temporary for the rounding addition. The stored activation retains eight fractional bits.

Pooling reads four activations through a synchronous memory port, retaining the maximum. There are two states per tap, READ and COMPARE, followed by WRITE. Its 392 outputs use the same channel-major order expected by the dense weights.

The dense layer processes 392 terms for each of 16 scores. Argmax consumes these signed scores in class order, beginning with class 0 even if every score is negative. A strict greater-than comparison preserves the lowest index on an exact tie.

## Input and completion interface

All inputs are synchronous to clk. Reset is active high and must be sampled on a rising edge.

| Signal | Meaning |
|---|---|
| pixel_valid / pixel_ready | A pixel byte is accepted on a rising edge when both are high |
| pixel_data[7:0] | Next byte in row-major 16x16 order |
| frame_ready | All 256 bytes have been accepted and the core is idle |
| start | Request inference; accepted only when frame_ready is high |
| busy | Computation is in progress |
| done | One-clock pulse when the result becomes available |
| prediction[3:0] | Class index, ordered 0123456789ABCDEF |
| cycle_count[31:0] | Clocks from accepted start to completion; retained until the next start |

A typical transaction is:

1. Assert reset for at least one rising edge, then release it.
2. Send 256 bytes using pixel_valid / pixel_ready. Gaps between bytes are allowed.
3. Observe frame_ready and pulse start for one cycle.
4. Wait for done; read prediction and cycle_count.
5. Load the next complete frame.

Start with a partial frame is ignored. A start pulse coincident with the 256th accepted byte is also too early: frame_ready was still low before that edge.

Extra bytes after a full frame and all bytes/start requests during computation are ignored. There is no buffering for a second image while busy.

The prediction remains available until another inference completes or reset is asserted. The controller accepts a new frame after completion.

Reset aborts any active inference and requires all 256 input bytes to be loaded again. Large arrays are intentionally not reset, preserving RAM inference. This is safe because the complete input and every needed intermediate value are rewritten before use.

The board wrapper in Step 5 must synchronize external inputs and provide the UART transaction control. Asynchronous buttons must not directly drive start or pixel_valid.

## Debug traces and testbenches

The core exposes three valid/data trace interfaces:

- conv_valid: conv_address, conv_accumulator, and conv_activation.
- pool_valid: pool_address and pool_value.
- score_valid: score_index and score_value.

Values are meaningful only when the corresponding valid signal is high. The testbench samples them on the active edge before the controller advances. A future board wrapper can leave trace outputs disconnected.

[tb_fixed_arithmetic.sv](../tb/tb_fixed_arithmetic.sv) checks 12 activation boundaries, three MAC scenarios, two argmax scenarios, and reset. These include signed extrema, full-width products, a worst-case 392-term accumulator, all-negative scores, and ties.

[tb_hexsight_cnn.sv](../tb/tb_hexsight_cnn.sv) compares every convolution accumulator, activation, pooled feature, score, and final class against the frozen [Step 3 vectors](../test_vectors/cnn-fixed-point/README.md).

The regression completes 25 inferences: all 24 distinct cases and a repeat of case 0 after all the others, without reset. It compares **88,625 intermediate/output words** exactly. The first 24 cases account for 85,080 of those words.

Additional checks cover input gaps, premature starts, extra pixels, requests while busy, one-cycle completion, retained results, and three reset aborts: during convolution, pooling, and dense inference. The core must reload a complete frame after each abort.

Every mismatch, missing output, wrong address, incorrect latency, or timeout fails the run. The Python driver also checks the emitted reports and predictions, rather than relying only on the simulator's process exit code.

## Latency

The measured deterministic core latency is **69,433 clocks**:

| Work | Cycles |
|---|---:|
| Convolution: 1,568 outputs x (2 bias/load + 9 x 3 MAC + 1 write) | 47,040 |
| Pooling: 392 outputs x (4 x 2 read/compare + 1 write) | 3,528 |
| Dense: 16 scores x (2 bias/load + 392 x 3 MAC + 1 write) | 18,864 |
| Completion | 1 |
| Total | 69,433 |

At 100 MHz, that corresponds to **694.33 microseconds of core computation**. Input loading, UART communication, laptop processing, and display refresh are excluded. No FPGA-board latency or CPU speedup has been measured.

## Synthesis and timing scope

The target is the Nexys A7-100T's xc7a100tcsg324-1. The driver synthesizes the core, places it, and routes it with Vivado 2024.1.

[core_ooc.xdc](../constraints/core_ooc.xdc) defines a 10 ns clock and 0.1 ns clock uncertainty. It includes provisional 2 ns maximum and 0.5 ns minimum port delays. It does not assign board pins or define the final board interface.

Only internal register-to-register timing is the acceptance criterion for this standalone core. The script explicitly reports those paths and checks that there are no unclocked registers, unconstrained internal endpoints, combinational loops, or critical DRC errors.

The core's ports have no physical locations or surrounding logic in this build. Their timing is therefore not closed. Full timing reports and tool warnings are retained, including the reported boundary hold failures; no false-path constraints are added to hide them.

See [the measured RTL report](../results/rtl-inference/README.md) for resource counts, internal setup/hold slack, boundary limitations, and links to the original reports. Board-level timing must be checked again after integration.

## Reproduce Step 4

Use the existing project Python environment and Vivado installation. Raw EMNIST data is not needed for this regression; the checked-in weights and vectors are sufficient.

On this workstation, from the repository root:

    $HexSightPython = (Get-Content local.settings.json | ConvertFrom-Json).python
    & $HexSightPython python/run_rtl.py --results-dir results/rtl-reproduced

On another Windows machine:

    & .\.venv\Scripts\python.exe python/run_rtl.py --vivado 'C:\Xilinx\Vivado\2024.1\bin\vivado.bat' --results-dir results/rtl-reproduced

For simulation without synthesis/routing:

    & $HexSightPython python/run_rtl.py --mode sim --results-dir results/local/rtl-sim-new

The runner checks model/vector hashes, cross-checks the vectors in Python, and stages the sources and memory files into a fresh local temporary directory. This avoids Vivado batch-launcher problems with UNC network working directories.

It invokes xvlog, xelab, and xsim, then the Vivado batch Tcl flow. Tool console output stays in the printed build directory. Published result directories must be new or empty, so previous evidence is never overwritten.

To select a persistent local build location, pass --work-dir with a new or empty local-drive directory. The build directory retains diagnostic files after a failure.

The final verification.json records tool version, normalized source hashes, model/vector hashes, per-case checks, timing scope, and report hashes. RTL source hashes normalize CRLF to LF before staging, so line-ending differences do not change the recorded code identity.

## Next: Step 5, board integration

Add the Nexys A7 top-level wrapper, verified pin constraints, synchronized reset, UART receiver/transmitter and framing, inference control, and seven-segment multiplexing. Then run complete-design timing checks and program the board.

The current core has no UART, display driver, or bitstream. Its numerical behavior is verified in RTL simulation, and the standalone resource/timing evidence is a foundation for that integration.

## Tool references

- [AMD XSim batch elaboration flow](https://docs.amd.com/r/2024.1-English/ug900-vivado-logic-simulation/xelab)
- [AMD block-RAM read/write synchronization](https://docs.amd.com/r/2024.1-English/ug901-vivado-synthesis/Block-RAM-Read/Write-Synchronization-Modes)
- [AMD timing-path queries](https://docs.amd.com/r/2022.2-English/ug835-vivado-tcl-commands/get_timing_paths)
