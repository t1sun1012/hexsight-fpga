# Fixed-point CNN model for FPGA inference

Source: the verified Step 2 CNN, epoch 46. See [Step 3](../../docs/step-03.md) for the complete arithmetic contract.

| File | Shape | Signed width | Fraction bits |
|---|---|---:|---:|
| conv_weight.mem | 8 x 1 x 3 x 3 | 16 | 12 |
| conv_bias.mem | 8 | 32 | 20 |
| fc_weight.mem | 16 x 392 | 16 | 12 |
| fc_bias.mem | 16 | 32 | 20 |

Values are one uppercase hexadecimal word per line, in two's complement, with LF newlines. Weight words use four hex digits; biases use eight. Total parameter storage is 12,784 bytes.

Addresses:

    conv_weight: channel * 9 + kernel_row * 3 + kernel_column
    conv_bias:   channel
    fc_weight:   class * 392 + channel * 49 + pooled_row * 7 + pooled_column
    fc_bias:     class

Classes: 0123456789ABCDEF. Kernels use cross-correlation, with no flipping.

Use $readmemh to load these files into appropriately sized memories. Declare and propagate signed values explicitly, zero-extend unsigned input bytes, and sign-extend biases/products before accumulation. Width declarations alone do not guarantee that an entire Verilog expression is signed.

fixed_point.vh contains generated localparam declarations to include inside a module. It provides dimensions and precision, not the RTL datapath.

manifest.json is the machine-readable specification and SHA-256 inventory. python/verify_fixed_point.py checks it against both the exported files and the floating-point checkpoint. Output scores use 40 signed bits with 20 fractional bits; their argmax needs no floating-point conversion.

These parameters have been verified in software. RTL, synthesis, and board execution have not yet been completed.
