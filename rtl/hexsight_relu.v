`timescale 1ns/1ps
// Scale 2^20 -> 2^8, then ReLU/saturate. Positive half ties round upward.
module hexsight_relu (
    input wire signed [39:0] accumulator,
    output wire signed [15:0] activation
);
    wire [40:0] rounded = {1'b0, accumulator} + 41'd2048;
    assign activation = (accumulator <= 40'sd0) ? 16'sd0 :
                        (rounded[40:12] > 29'd32767) ? 16'sd32767 :
                        $signed({1'b0, rounded[26:12]});
endmodule
