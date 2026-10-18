`timescale 1ns / 1ps
// Runs every line of out/test_<split>.mem through mlp_relu6 and compares with the Python
// fixed-point class (bit-exact check) and the true label (accuracy).
// Line format: 7 x int16 hex inputs, python class, true label.
`ifndef DUT
`define DUT mlp_relu6  // iverilog -DDUT=mlp_relu6_np etc. for the parallel variants
`endif
module tb_mlp_relu6;
    parameter MEM_DIR   = "mem/A/";
    parameter TEST_FILE = "../out/test_A.mem";

    reg clk = 1'b0, rst_n = 1'b0, start = 1'b0;
    reg [111:0] x = 112'd0;
    wire [2:0] class_out;
    wire valid, busy;
    always #10 clk = ~clk;  // 50 MHz, as in the paper

`ifdef NETLIST  // post-implementation netlist: weights are baked in, no parameters
    mlp_relu6 dut (  // netlists are written with the top renamed to mlp_relu6
.clk(clk), .rst_n(rst_n), .start(start), .x(x),
                   .class_out(class_out), .valid(valid), .busy(busy));
`else
    `DUT #(.W1_FILE({MEM_DIR, "w1.mem"}), .B1_FILE({MEM_DIR, "b1.mem"}),
                .W2_FILE({MEM_DIR, "w2.mem"}), .B2_FILE({MEM_DIR, "b2.mem"}))
        dut (.clk(clk), .rst_n(rst_n), .start(start), .x(x),
             .class_out(class_out), .valid(valid), .busy(busy));
`endif

    integer fd, r, total, mismatches, correct, cycles, latency;
    reg [15:0] f0, f1, f2, f3, f4, f5, f6;
    reg [3:0] py, label;
    reg [8*256:1] tf;  // +TEST=<file> overrides TEST_FILE at run time
    initial begin
        total = 0; mismatches = 0; correct = 0; latency = 0;
        if (!$value$plusargs("TEST=%s", tf)) tf = TEST_FILE;
        fd = $fopen(tf, "r");
        if (fd == 0) begin $display("cannot open %0s", tf); $finish; end
        #200;  // gate-level sims hold every flop in global reset (GSR) for the first 100 ns
        @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);
        while (!$feof(fd)) begin
            r = $fscanf(fd, "%h %h %h %h %h %h %h %h %h\n", f0, f1, f2, f3, f4, f5, f6, py, label);
            if (r == 9) begin
                // non-blocking drives: the DUT samples the old values at this edge, so no race
                x <= {f6, f5, f4, f3, f2, f1, f0};
                start <= 1'b1; @(posedge clk); start <= 1'b0;
                cycles = 1;
                while (!valid) begin
                    @(posedge clk); cycles = cycles + 1;
                    if (cycles > 1000) begin $display("TIMEOUT at sample %0d", total); $display("RESULT: FAIL"); $finish; end
                end
                latency = cycles;
                if (class_out != py[2:0]) begin
                    mismatches = mismatches + 1;
                    if (mismatches <= 10)
                        $display("MISMATCH sample %0d: rtl=%0d python=%0d", total, class_out, py);
                end
                if (class_out == label[2:0]) correct = correct + 1;
                total = total + 1;
                @(posedge clk);
            end
        end
        $fclose(fd);
        $display("%0s: samples=%0d mismatches_vs_python=%0d rtl_accuracy=%0d/%0d latency_cycles=%0d",
                 tf, total, mismatches, correct, total, latency);
        if (total == 0 || mismatches != 0) $display("RESULT: FAIL");
        else $display("RESULT: PASS (bit-exact)");
        $finish;
    end
endmodule
