`timescale 1ns/1ps
// Assert multiply_en, then accumulate_en on a later clock. Load overrides add.
module hexsight_mac (
    input wire clk, input wire rst,
    input wire load_en, input wire signed [39:0] load_value,
    input wire multiply_en,
    input wire signed [15:0] operand_a, input wire signed [15:0] operand_b,
    input wire accumulate_en,
    output reg signed [39:0] accumulator
);
    (* use_dsp = "yes" *) reg signed [31:0] product;
    wire signed [39:0] extended_product = {{8{product[31]}}, product};
    always @(posedge clk) begin
        if (rst) begin
            product <= 32'sd0;
            accumulator <= 40'sd0;
        end else begin
            if (multiply_en)
                product <= operand_a * operand_b;
            if (load_en)
                accumulator <= load_value;
            else if (accumulate_en)
                accumulator <= accumulator + extended_product;
        end
    end
endmodule
