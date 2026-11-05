`timescale 1ns/1ps
module tb_fixed_arithmetic;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, load_en = 0, multiply_en = 0, accumulate_en = 0;
    reg signed [39:0] load_value = 0, relu_input = 0, score = 0;
    reg signed [15:0] operand_a = 0, operand_b = 0;
    wire signed [39:0] accumulator, best_score;
    wire signed [15:0] activation;
    reg valid = 0;
    reg [3:0] index = 0;
    wire [3:0] best_index;
    integer i, relu_checks = 0, mac_checks = 0, argmax_checks = 0, report;
    hexsight_mac mac (.*);
    hexsight_relu relu (.accumulator(relu_input), .activation(activation));
    hexsight_argmax argmax (.*);

    task automatic check_relu(input logic signed [39:0] value, input logic signed [15:0] expected);
        relu_input = value;
        #1;
        if (activation !== expected)
            $fatal(1, "ReLU: input=%0d actual=%0d expected=%0d", value, activation, expected);
        relu_checks++;
    endtask
    task automatic load_acc(input logic signed [39:0] value);
        @(negedge clk); load_value = value; load_en = 1;
        @(negedge clk); load_en = 0;
    endtask
    task automatic add_product(input logic signed [15:0] a, input logic signed [15:0] b);
        @(negedge clk); operand_a = a; operand_b = b; multiply_en = 1;
        @(negedge clk); multiply_en = 0; accumulate_en = 1;
        @(negedge clk); accumulate_en = 0;
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst = 0;
        check_relu(-40'sd549755813888, 0);
        check_relu(-40'sd2048, 0);
        check_relu(0, 0);
        check_relu(2047, 0);
        check_relu(2048, 1);
        check_relu(6143, 1);
        check_relu(6144, 2);
        check_relu(134211583, 32766);
        check_relu(134211584, 32767);
        check_relu(134215679, 32767);
        check_relu(134215680, 32767);
        check_relu(40'sd549755813887, 32767);

        load_acc(-40'sd2147483648);
        add_product(16'sd255, -16'sd32768);
        if (accumulator !== -40'sd2155839488) $fatal(1, "Signed pixel/weight/bias MAC");
        mac_checks++;
        load_acc(0);
        add_product(-16'sd32768, -16'sd32768);
        if (accumulator !== 40'sd1073741824) $fatal(1, "Full signed multiplier width");
        mac_checks++;
        load_acc(-40'sd2147483648);
        for (i = 0; i < 392; i++)
            add_product(16'sd32767, -16'sd32768);
        if (accumulator !== -40'sd423041433600) $fatal(1, "Dense accumulation truncated");
        mac_checks++;

        for (i = 0; i < 16; i++) begin
            @(negedge clk); valid = 1; index = i;
            score = ((i == 3) || (i == 7)) ? -40'sd2 : -40'sd50;
            @(negedge clk); valid = 0;
        end
        if (best_index !== 3 || best_score !== -40'sd2) $fatal(1, "Negative argmax/tie");
        argmax_checks++;
        @(negedge clk); valid = 1; index = 0; score = -40'sd549755813888;
        @(negedge clk); index = 1; score = 40'sd549755813887;
        @(negedge clk); valid = 0;
        if (best_index !== 1 || best_score !== 40'sd549755813887) $fatal(1, "Signed score extremes");
        argmax_checks++;
        @(negedge clk); rst = 1;
        @(negedge clk);
        if (accumulator !== 0 || best_index !== 0 || best_score !== 0) $fatal(1, "Arithmetic reset");
        report = $fopen("arithmetic_results.json", "w");
        if (!report) $fatal(1, "Cannot open arithmetic result");
        $fwrite(report, "{\"relu_checks\":%0d,\"mac_checks\":%0d,\"argmax_checks\":%0d,\"reset_checks\":1,\"passed\":true}\n",
                relu_checks, mac_checks, argmax_checks);
        $fclose(report);
        $display("ARITHMETIC_PASS");
        $finish;
    end
    initial begin
        #100000;
        $fatal(1, "Arithmetic timeout");
    end
endmodule
