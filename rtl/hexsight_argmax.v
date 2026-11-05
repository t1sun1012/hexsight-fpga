`timescale 1ns/1ps
// Consume scores in order 0..15; gaps are allowed. Index zero starts a new run.
// Strict > gives the lowest index on ties, including all-negative scores.
module hexsight_argmax (
    input wire clk, input wire rst,
    input wire valid, input wire [3:0] index,
    input wire signed [39:0] score,
    output reg [3:0] best_index,
    output reg signed [39:0] best_score
);
    always @(posedge clk) begin
        if (rst) begin
            best_index <= 4'd0;
            best_score <= 40'sd0;
        end else if (valid) begin
            if ((index == 4'd0) || (score > best_score)) begin
                best_index <= index;
                best_score <= score;
            end
        end
    end
endmodule
