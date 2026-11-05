`timescale 1ns/1ps
// 16x16 -> Conv(8,3x3) -> ReLU -> MaxPool(2) -> FC(16).
// Input bytes arrive sequentially. Start is accepted only for a full frame.
// No array reset: reset discards the frame; all required words are rewritten.
module hexsight_cnn #(
    parameter CONV_WEIGHT_FILE = "conv_weight.mem",
    parameter CONV_BIAS_FILE = "conv_bias.mem",
    parameter FC_WEIGHT_FILE = "fc_weight.mem",
    parameter FC_BIAS_FILE = "fc_bias.mem"
) (
    input wire clk, input wire rst,
    input wire pixel_valid, output wire pixel_ready, input wire [7:0] pixel_data,
    input wire start, output wire frame_ready,
    output wire busy, output reg done,
    output reg [3:0] prediction, output reg [31:0] cycle_count,
    output wire conv_valid, output wire [10:0] conv_address,
    output wire signed [39:0] conv_accumulator,
    output wire signed [15:0] conv_activation,
    output wire pool_valid, output wire [8:0] pool_address,
    output wire signed [15:0] pool_value,
    output wire score_valid, output wire [3:0] score_index,
    output wire signed [39:0] score_value
);
    `include "fixed_point.vh"

    localparam [4:0]
        IDLE = 0, CONV_BIAS_READ = 1, CONV_INIT = 2,
        CONV_READ = 3, CONV_MUL = 4, CONV_ADD = 5, CONV_WRITE = 6,
        POOL_READ = 7, POOL_COMPARE = 8, POOL_WRITE = 9,
        FC_BIAS_READ = 10, FC_INIT = 11, FC_READ = 12,
        FC_MUL = 13, FC_ADD = 14, FC_WRITE = 15, FINISH = 16;
    reg [4:0] state;
    reg [8:0] pixel_count;
    reg [2:0] channel;
    reg [3:0] conv_row, conv_col;
    reg [1:0] kernel_row, kernel_col;
    reg [10:0] conv_write_address;
    reg [2:0] pool_row, pool_col, pool_channel;
    reg [1:0] pool_tap;
    reg [8:0] pool_write_address;
    reg [3:0] output_class;
    reg [8:0] feature_index;
    reg [12:0] fc_weight_address;

    (* ram_style = "distributed" *) reg [7:0] pixels [0:255];
    (* rom_style = "distributed" *) reg signed [15:0] conv_weights [0:71];
    (* rom_style = "distributed" *) reg signed [31:0] conv_biases [0:7];
    (* rom_style = "block" *) reg signed [15:0] fc_weights [0:6271];
    (* rom_style = "distributed" *) reg signed [31:0] fc_biases [0:15];
    (* ram_style = "block" *) reg signed [15:0] activations [0:1567];
    (* ram_style = "block" *) reg signed [15:0] pooled [0:391];

    reg [7:0] pixel_q;
    reg signed [15:0] conv_weight_q, fc_weight_q, activation_q, pooled_q;
    reg signed [31:0] conv_bias_q, fc_bias_q;
    reg signed [15:0] pool_max;

    wire [3:0] input_row = conv_row + kernel_row;
    wire [3:0] input_col = conv_col + kernel_col;
    wire [7:0] pixel_read_address = {input_row, input_col};
    wire [6:0] conv_weight_address = channel * 9 + kernel_row * 3 + kernel_col;
    wire [10:0] activation_read_address =
        pool_channel * 196 + pool_row * 28 + pool_col * 2 + pool_tap[1] * 14 + pool_tap[0];

    wire mac_load = (state == CONV_INIT) || (state == FC_INIT);
    wire mac_multiply = (state == CONV_MUL) || (state == FC_MUL);
    wire mac_add = (state == CONV_ADD) || (state == FC_ADD);
    wire signed [39:0] mac_load_value = (state == CONV_INIT) ?
        $signed({{8{conv_bias_q[31]}}, conv_bias_q}) :
        $signed({{8{fc_bias_q[31]}}, fc_bias_q});
    wire signed [15:0] mac_a = (state == CONV_MUL) ? $signed({8'b0, pixel_q}) : pooled_q;
    wire signed [15:0] mac_b = (state == CONV_MUL) ? conv_weight_q : fc_weight_q;
    wire signed [39:0] mac_accumulator;
    wire signed [15:0] requantized;
    wire [3:0] best_index;
    wire signed [39:0] best_score;

    hexsight_mac mac (
        .clk(clk), .rst(rst), .load_en(mac_load), .load_value(mac_load_value),
        .multiply_en(mac_multiply), .operand_a(mac_a), .operand_b(mac_b),
        .accumulate_en(mac_add), .accumulator(mac_accumulator)
    );
    hexsight_relu relu (.accumulator(mac_accumulator), .activation(requantized));
    hexsight_argmax argmax (
        .clk(clk), .rst(rst), .valid(score_valid), .index(output_class),
        .score(mac_accumulator), .best_index(best_index), .best_score(best_score)
    );

    assign pixel_ready = !rst && (state == IDLE) && (pixel_count < 9'd256);
    assign frame_ready = !rst && (state == IDLE) && (pixel_count == 9'd256);
    assign busy = !rst && (state != IDLE);
    assign conv_valid = !rst && (state == CONV_WRITE);
    assign conv_address = conv_write_address;
    assign conv_accumulator = mac_accumulator;
    assign conv_activation = requantized;
    assign pool_valid = !rst && (state == POOL_WRITE);
    assign pool_address = pool_write_address;
    assign pool_value = pool_max;
    assign score_valid = !rst && (state == FC_WRITE);
    assign score_index = output_class;
    assign score_value = mac_accumulator;

    initial begin
        $readmemh(CONV_WEIGHT_FILE, conv_weights);
        $readmemh(CONV_BIAS_FILE, conv_biases);
        $readmemh(FC_WEIGHT_FILE, fc_weights);
        $readmemh(FC_BIAS_FILE, fc_biases);
    end

    // Synchronous reads; the following MUL/COMPARE state consumes the data.
    always @(posedge clk) begin
        if (pixel_valid && pixel_ready)
            pixels[pixel_count[7:0]] <= pixel_data;
        if (state == CONV_BIAS_READ)
            conv_bias_q <= conv_biases[channel];
        if (state == CONV_READ) begin
            pixel_q <= pixels[pixel_read_address];
            conv_weight_q <= conv_weights[conv_weight_address];
        end
        if (state == CONV_WRITE)
            activations[conv_write_address] <= requantized;
        if (state == POOL_READ)
            activation_q <= activations[activation_read_address];
        if (state == POOL_WRITE)
            pooled[pool_write_address] <= pool_max;
        if (state == FC_BIAS_READ)
            fc_bias_q <= fc_biases[output_class];
        if (state == FC_READ) begin
            pooled_q <= pooled[feature_index];
            fc_weight_q <= fc_weights[fc_weight_address];
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            pixel_count <= 0;
            channel <= 0;
            conv_row <= 0;
            conv_col <= 0;
            kernel_row <= 0;
            kernel_col <= 0;
            conv_write_address <= 0;
            pool_channel <= 0;
            pool_row <= 0;
            pool_col <= 0;
            pool_tap <= 0;
            pool_write_address <= 0;
            pool_max <= 0;
            output_class <= 0;
            feature_index <= 0;
            fc_weight_address <= 0;
            done <= 0;
            prediction <= 0;
            cycle_count <= 0;
        end else begin
            done <= 0;
            if (state != IDLE)
                cycle_count <= cycle_count + 1'b1;
            case (state)
                IDLE: begin
                    if (pixel_valid && pixel_ready)
                        pixel_count <= pixel_count + 1'b1;
                    if (start && frame_ready) begin
                        channel <= 0;
                        conv_row <= 0;
                        conv_col <= 0;
                        conv_write_address <= 0;
                        cycle_count <= 0;
                        state <= CONV_BIAS_READ;
                    end
                end
                CONV_BIAS_READ: state <= CONV_INIT;
                CONV_INIT: begin
                    kernel_row <= 0;
                    kernel_col <= 0;
                    state <= CONV_READ;
                end
                CONV_READ: state <= CONV_MUL;
                CONV_MUL: state <= CONV_ADD;
                CONV_ADD: begin
                    if (kernel_col == 2) begin
                        kernel_col <= 0;
                        if (kernel_row == 2)
                            state <= CONV_WRITE;
                        else begin
                            kernel_row <= kernel_row + 1'b1;
                            state <= CONV_READ;
                        end
                    end else begin
                        kernel_col <= kernel_col + 1'b1;
                        state <= CONV_READ;
                    end
                end
                CONV_WRITE: begin
                    if (conv_write_address == 1567) begin
                        pool_channel <= 0;
                        pool_row <= 0;
                        pool_col <= 0;
                        pool_tap <= 0;
                        pool_write_address <= 0;
                        state <= POOL_READ;
                    end else begin
                        conv_write_address <= conv_write_address + 1'b1;
                        if (conv_col == 13) begin
                            conv_col <= 0;
                            if (conv_row == 13) begin
                                conv_row <= 0;
                                channel <= channel + 1'b1;
                            end else
                                conv_row <= conv_row + 1'b1;
                        end else
                            conv_col <= conv_col + 1'b1;
                        state <= CONV_BIAS_READ;
                    end
                end
                POOL_READ: state <= POOL_COMPARE;
                POOL_COMPARE: begin
                    if ((pool_tap == 0) || (activation_q > pool_max))
                        pool_max <= activation_q;
                    if (pool_tap == 3)
                        state <= POOL_WRITE;
                    else begin
                        pool_tap <= pool_tap + 1'b1;
                        state <= POOL_READ;
                    end
                end
                POOL_WRITE: begin
                    pool_tap <= 0;
                    if (pool_write_address == 391) begin
                        output_class <= 0;
                        fc_weight_address <= 0;
                        state <= FC_BIAS_READ;
                    end else begin
                        pool_write_address <= pool_write_address + 1'b1;
                        if (pool_col == 6) begin
                            pool_col <= 0;
                            if (pool_row == 6) begin
                                pool_row <= 0;
                                pool_channel <= pool_channel + 1'b1;
                            end else
                                pool_row <= pool_row + 1'b1;
                        end else
                            pool_col <= pool_col + 1'b1;
                        state <= POOL_READ;
                    end
                end
                FC_BIAS_READ: state <= FC_INIT;
                FC_INIT: begin
                    feature_index <= 0;
                    state <= FC_READ;
                end
                FC_READ: state <= FC_MUL;
                FC_MUL: state <= FC_ADD;
                FC_ADD: begin
                    fc_weight_address <= fc_weight_address + 1'b1;
                    if (feature_index == 391)
                        state <= FC_WRITE;
                    else begin
                        feature_index <= feature_index + 1'b1;
                        state <= FC_READ;
                    end
                end
                FC_WRITE: begin
                    if (output_class == 15)
                        state <= FINISH;
                    else begin
                        output_class <= output_class + 1'b1;
                        state <= FC_BIAS_READ;
                    end
                end
                FINISH: begin
                    prediction <= best_index;
                    done <= 1;
                    pixel_count <= 0;
                    state <= IDLE;
                end
                default: begin
                    state <= IDLE;
                    pixel_count <= 0;
                end
            endcase
        end
    end
endmodule
