# Step 4 RTL verification and implementation report

Completed November 5, 2026 with Vivado 2024.1 for xc7a100tcsg324-1.

**The Verilog CNN matches the integer reference exactly on the exported regression cases. Internal routed timing passes at 100 MHz. Standalone boundary timing is not closed, and the FPGA board has not been programmed.**

## Functional verification

| Check | Result |
|---|---:|
| Distinct reference cases | 24 |
| Completed inferences, including one repeat | 25 |
| Intermediate/output words compared | 88,625 |
| Numerical mismatches | 0 |
| Reset abort/recovery scenarios | 3 |
| Activation boundary checks | 12 |
| Signed MAC scenarios | 3 |
| Argmax scenarios | 2 |
| Arithmetic reset check | 1 |
| Existing Python tests | 26 passed |

The full-CNN test compares all convolution accumulators, rounded activations, pooled features, dense scores, and final predictions. It also checks input gaps, premature starts, writes to a full input frame, requests while busy, completion pulse width, retained outputs, and repeated inference without reset.

Reset scenarios interrupt convolution, pooling, and dense inference. A complete new frame is required afterward.

These are arithmetic/control regression cases, not an independent accuracy evaluation. The full validation accuracy belongs to the [Step 3 Python evaluation](../fixed-point/README.md); the official test set remains unused.

## Latency

Every completed inference takes **69,433 clocks** from accepted start to done.

At 100 MHz this corresponds to **694.33 microseconds** of core computation. It excludes input loading, UART, laptop preprocessing, and display activity. It is derived from simulation cycles and the clock target, not measured on a board or compared against CPU inference.

## Routed resources

The standalone core uses:

| Resource | Used | Available on target |
|---|---:|---:|
| DSP48E1 | 1 | 240 |
| RAMB36E1 | 5 | 135 |
| RAMB18E1 | 1 | 270 half-blocks |
| Block-RAM tile equivalents | 5.5 | 135 |
| Packed slice LUTs | 384 | 63,400 |
| Slice registers | 244 | 126,800 |
| Latches | 0 | â€” |

The RAMB18 and RAMB36 capacities share the same physical block-RAM resources; they are not additive independent capacities.

Of the 384 packed LUTs, 336 implement logic and 48 implement distributed memory. synthesis_results.json also records 480 logic LUT primitives before accounting for physical LUT packing; use the packed report for occupied slice-LUT resources.

The one DSP implements the shared multiplier/accumulator. The dense weights, convolution activations, and pooled features use block RAM. Pixels and small lookup tables use distributed resources.

See [routed_utilization.rpt](routed_utilization.rpt) and [synthesis_utilization.rpt](synthesis_utilization.rpt). These include trace outputs; a final board wrapper may change optimization and resource use.

## Timing and its limits

Clock period: 10.000 ns. User clock uncertainty: 0.100 ns.

| Timing scope | Worst setup slack | Worst hold slack |
|---|---:|---:|
| Internal register-to-register paths | +3.094 ns | +0.057 ns |
| All paths with provisional port delays | +0.504 ns | -0.687 ns |

The acceptance check for this core-only build is internal register-to-register timing. It includes sequential memory endpoints. There are no unclocked registers, unconstrained internal endpoints, or combinational loops in check_timing.rpt.

**The full design is not timing-closed.** The negative hold slack is on unlocated boundary paths, including reset and pixel inputs. Vivado warns that timing to/from these ports is not accurate without physical partition locations or surrounding logic. The provisional port delays are 2.0 ns maximum and 0.5 ns minimum; they are not measurements of the future UART wrapper.

No false paths were added to mask those failures. Both internal and full reports remain available:

- [Internal timing paths](internal_timing.rpt)
- [Full timing summary, including boundary hold failures](routed_timing.rpt)
- [Constraint completeness check](check_timing.rpt)
- [DRC report](routed_drc.rpt)
- [Tool warnings](tool_warnings.txt)

There are no Error or Critical Warning DRC violations. The retained warnings include missing board configuration-voltage properties, DSP input-pipeline suggestions, unconnected standalone outputs, unplaced boundary ports, the boundary hold failures, and a tile-splitting warning. The constant-zero activation sign bit is expected after ReLU. Parallel-synthesis eligibility and the empty critical-DRC query also produce benign tool messages.

Step 5 must replace standalone assumptions with the real clock/reset/UART/display wrapper, verify board properties and pins, and rerun full setup/hold and DRC checks. This report does not establish a maximum operating frequency or board-level timing closure.

## Reproduction and provenance

Run from the repository root with the project Python environment:

    python python/run_rtl.py --results-dir results/rtl-reproduced

Use --vivado to specify the Vivado launcher, --mode sim for simulation only, or --work-dir for a new local build directory. See [the Step 4 guide](../../docs/step-04.md) for PowerShell commands and interface details.

[verification.json](verification.json) binds the source hashes, exported model, test vectors, simulator results, timing metrics, and report hashes. Source hashes normalize CRLF to LF. Tool reports retain their exact bytes.

[rtl_cases.csv](rtl_cases.csv) records every checked prediction and latency. arithmetic_results.json, cnn_results.json, and synthesis_results.json provide the corresponding machine-readable summaries.
