`timescale 1ns / 1ps
// 7-6-5 ReLU6 MLP, fully parallel: all 42 + 30 products in one cycle per layer (72 multipliers).
// Products are registered (DSP MREG) before each adder tree so 50 MHz closes.
// Same ports, formats and results as mlp_relu6.v (bit-exact). Cycles start -> valid: 7.
module mlp_relu6_par #(
    parameter F1 = 12,
    parameter F2 = 13,
    parameter W1_FILE = "w1.mem",
    parameter B1_FILE = "b1.mem",
    parameter W2_FILE = "w2.mem",
    parameter B2_FILE = "b2.mem"
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [111:0] x,
    output reg  [2:0]   class_out,
    output reg          valid,
    output wire         busy
);
    localparam signed [31:0] RELU6_MAX = 32'sd24576;

    reg signed [15:0] w1 [0:41];
    reg signed [31:0] b1 [0:5];
    reg signed [15:0] w2 [0:29];
    reg signed [31:0] b2 [0:4];
    initial begin
        $readmemh(W1_FILE, w1);
        $readmemh(B1_FILE, b1);
        $readmemh(W2_FILE, w2);
        $readmemh(B2_FILE, b2);
    end

    reg signed [15:0] xr [0:6];
    reg signed [15:0] h  [0:5];
    reg signed [31:0] z  [0:4];
    reg signed [31:0] p1 [0:41], p2 [0:29];   // registered products
    reg [4:0] pipe;                // valid bits: x, p1, h, p2, z
    assign busy = |pipe;

    reg signed [31:0] a1 [0:5], a2 [0:4], shr, best;
    reg [2:0] cls;
    integer n, k;
    always @* begin
        for (n = 0; n < 6; n = n + 1) begin
            a1[n] = b1[n];
            for (k = 0; k < 7; k = k + 1) a1[n] = a1[n] + p1[k * 6 + n];
        end
        for (n = 0; n < 5; n = n + 1) begin
            a2[n] = b2[n];
            for (k = 0; k < 6; k = k + 1) a2[n] = a2[n] + p2[k * 5 + n];
        end
        best = z[0]; cls = 3'd0;
        for (n = 1; n < 5; n = n + 1)
            if (z[n] > best) begin best = z[n]; cls = n; end
    end

    always @(posedge clk) begin
        if (start) for (n = 0; n < 7; n = n + 1) xr[n] <= x[16 * n +: 16];
        if (pipe[0]) for (k = 0; k < 42; k = k + 1) p1[k] <= xr[k / 6] * w1[k];
        if (pipe[1]) for (n = 0; n < 6; n = n + 1) begin
            shr = a1[n] >>> F1;
            h[n] <= (shr < 0) ? 16'sd0 : (shr > RELU6_MAX) ? RELU6_MAX[15:0] : shr[15:0];
        end
        if (pipe[2]) for (k = 0; k < 30; k = k + 1) p2[k] <= h[k / 5] * w2[k];
        if (pipe[3]) for (n = 0; n < 5; n = n + 1) z[n] <= a2[n];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe <= 5'd0; valid <= 1'b0; class_out <= 3'd0;
        end else begin
            pipe <= {pipe[3:0], start};
            valid <= pipe[4];
            if (pipe[4]) class_out <= cls;
        end
    end
endmodule
