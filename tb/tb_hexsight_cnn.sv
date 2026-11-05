`timescale 1ns/1ps
module tb_hexsight_cnn;
    localparam CASES = 24;
    localparam EXPECTED_CYCLES = 69433;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, pixel_valid = 0, start = 0;
    reg [7:0] pixel_data = 0;
    wire pixel_ready, frame_ready, busy, done, conv_valid, pool_valid, score_valid;
    wire [3:0] prediction, score_index;
    wire [31:0] cycle_count;
    wire [10:0] conv_address;
    wire [8:0] pool_address;
    wire signed [39:0] conv_accumulator, score_value;
    wire signed [15:0] conv_activation, pool_value;
    hexsight_cnn dut (.*);

    reg [7:0] expected_pixels [0:CASES*256-1];
    reg signed [39:0] expected_conv [0:CASES*1568-1];
    reg signed [15:0] expected_relu [0:CASES*1568-1];
    reg signed [15:0] expected_pool [0:CASES*392-1];
    reg signed [39:0] expected_logits [0:CASES*16-1];
    reg [3:0] expected_predictions [0:CASES-1];
    integer active_case = 0, conv_count = 0, pool_count = 0, score_count = 0;
    integer completed = 0, reset_aborts = 0, total_words = 0, report, summary, c;
    bit checking = 0;
    reg [3:0] held_prediction;

    // Sample valid/data immediately before each active edge; no NBA race.
    always @(posedge clk) begin
        if (!rst && checking) begin
            if (conv_valid) begin
                if (conv_count >= 1568 || conv_address !== conv_count[10:0])
                    $fatal(1, "Convolution count/address case=%0d index=%0d", active_case, conv_count);
                if (conv_accumulator !== expected_conv[active_case*1568+conv_count])
                    $fatal(1, "Conv mismatch case=%0d addr=%0d actual=%0d expected=%0d",
                           active_case, conv_count, conv_accumulator, expected_conv[active_case*1568+conv_count]);
                if (conv_activation !== expected_relu[active_case*1568+conv_count])
                    $fatal(1, "ReLU mismatch case=%0d addr=%0d", active_case, conv_count);
                conv_count++;
            end
            if (pool_valid) begin
                if (conv_count != 1568 || pool_count >= 392 || pool_address !== pool_count[8:0])
                    $fatal(1, "Pool order/count");
                if (pool_value !== expected_pool[active_case*392+pool_count])
                    $fatal(1, "Pool mismatch case=%0d addr=%0d", active_case, pool_count);
                pool_count++;
            end
            if (score_valid) begin
                if (pool_count != 392 || score_count >= 16 || score_index !== score_count[3:0])
                    $fatal(1, "Score order/count");
                if (score_value !== expected_logits[active_case*16+score_count])
                    $fatal(1, "Score mismatch case=%0d class=%0d actual=%0d expected=%0d",
                           active_case, score_count, score_value, expected_logits[active_case*16+score_count]);
                score_count++;
            end
            if (busy && (pixel_ready || frame_ready))
                $fatal(1, "Input must be blocked during inference");
        end
    end

    task automatic reset_core;
        @(negedge clk); checking = 0; rst = 1; start = 0; pixel_valid = 0;
        repeat (2) @(negedge clk);
        if (done !== 0 || busy !== 0 || frame_ready !== 0 || prediction !== 0)
            $fatal(1, "Reset did not discard frame/result");
        rst = 0;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        if (busy !== 0 || frame_ready !== 0 || pixel_ready !== 1)
            $fatal(1, "Reset allowed stale frame to start");
    endtask

    task automatic upload(input integer case_number, input bit exercise_protocol);
        integer i;
        for (i = 0; i < 256; i++) begin
            @(negedge clk);
            if (!pixel_ready) $fatal(1, "Input not ready at pixel %0d", i);
            pixel_valid = 1;
            pixel_data = expected_pixels[case_number*256+i];
            // Pulse start with a partial frame and with the final input byte.
            start = exercise_protocol && ((i == 100) || (i == 255));
            if (exercise_protocol && (i % 17 == 16)) begin
                @(negedge clk); pixel_valid = 0; start = 0;
            end
        end
        @(negedge clk); pixel_valid = 0; start = 0;
        if (frame_ready !== 1 || pixel_ready !== 0 || busy !== 0)
            $fatal(1, "Frame completion / early start protocol");
        // Extra writes while the frame is full must not corrupt its contents.
        if (exercise_protocol) begin
            pixel_valid = 1; pixel_data = 8'hA5;
            repeat (2) @(negedge clk);
            pixel_valid = 0;
        end
    endtask

    task automatic start_core;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        if (busy !== 1) $fatal(1, "Complete frame did not start");
    endtask

    task automatic run_case(input integer case_number);
        integer timeout_cycles;
        upload(case_number, 1);
        active_case = case_number;
        conv_count = 0; pool_count = 0; score_count = 0; checking = 1;
        start_core();
        // Requests while busy must be ignored.
        start = 1; pixel_valid = 1; pixel_data = 8'hFF;
        repeat (2) @(negedge clk);
        start = 0; pixel_valid = 0;
        timeout_cycles = 0;
        while (!done && timeout_cycles < EXPECTED_CYCLES + 10) begin
            @(negedge clk); timeout_cycles++;
        end
        if (done !== 1 || busy !== 0) $fatal(1, "Inference timeout/completion");
        if (conv_count != 1568 || pool_count != 392 || score_count != 16)
            $fatal(1, "Missing stage outputs");
        if (prediction !== expected_predictions[case_number])
            $fatal(1, "Prediction mismatch case=%0d", case_number);
        if (cycle_count !== EXPECTED_CYCLES)
            $fatal(1, "Latency actual=%0d expected=%0d", cycle_count, EXPECTED_CYCLES);
        total_words += 3545;
        completed++;
        $fwrite(report, "%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                completed, case_number, prediction, cycle_count, conv_count, pool_count, score_count);
        $display("CASE_PASS case=%0d predicted=%h cycles=%0d", case_number, prediction, cycle_count);
        held_prediction = prediction;
        checking = 0;
        repeat (3) begin
            @(negedge clk);
            if (done !== 0 || prediction !== held_prediction || cycle_count !== EXPECTED_CYCLES)
                $fatal(1, "Completion must pulse once and retain result/cycle count");
        end
    endtask

    initial begin
        $readmemh("vectors_pixels.mem", expected_pixels);
        $readmemh("vectors_conv_acc.mem", expected_conv);
        $readmemh("vectors_relu.mem", expected_relu);
        $readmemh("vectors_pool.mem", expected_pool);
        $readmemh("vectors_logits.mem", expected_logits);
        $readmemh("vectors_predictions.mem", expected_predictions);
        report = $fopen("rtl_cases.csv", "w");
        if (!report) $fatal(1, "Cannot open case report");
        $fwrite(report, "run,case_index,prediction,cycles,conv_outputs,pool_outputs,scores\n");
        reset_core();

        // Abort in convolution, pooling, and dense inference, then require reload.
        upload(0, 0); start_core();
        repeat (100) @(negedge clk);
        reset_core(); reset_aborts++;
        upload(0, 0); start_core();
        wait (pool_valid); reset_core(); reset_aborts++;
        upload(0, 0); start_core();
        wait (score_valid); reset_core(); reset_aborts++;

        for (c = 0; c < CASES; c++) run_case(c);
        run_case(0); // Same image after all others, with no reset.
        $fclose(report);
        summary = $fopen("cnn_results.json", "w");
        if (!summary) $fatal(1, "Cannot open summary");
        $fwrite(summary, "{\"unique_cases\":24,\"completed_inferences\":%0d,\"reset_abort_checks\":%0d,\"words_compared\":%0d,\"cycles_per_inference\":%0d,\"passed\":true}\n",
                completed, reset_aborts, total_words, EXPECTED_CYCLES);
        $fclose(summary);
        $display("CNN_PASS cases=%0d words=%0d reset_aborts=%0d", completed, total_words, reset_aborts);
        $finish;
    end
    initial begin
        #30000000;
        $fatal(1, "Global CNN simulation timeout");
    end
endmodule
