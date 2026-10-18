`timescale 1ns / 1ps
// 7-6-5 ReLU6 MLP, neuron-parallel: one multiply-accumulate per neuron (6 in layer 1, 5 in layer 2).
// Same ports, formats and results as mlp_relu6.v (bit-exact). Cycles start -> valid: 18.
module mlp_relu6_np #(
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
    reg signed [31:0] acc [0:5];   // layer 2 uses acc[0..4]

    localparam IDLE = 2'd0, L1 = 2'd1, L2 = 2'd2, OUT = 2'd3;
    reg [1:0] state;
    reg [2:0] k;                   // 0 = load biases, 1..K = accumulate input k-1
    assign busy = (state != IDLE);

    // per-neuron products for the current input
    reg signed [31:0] mac [0:5];
    reg signed [31:0] shr;
    integer n;
    always @* begin
        for (n = 0; n < 6; n = n + 1)
            if (state == L1) mac[n] = acc[n] + xr[k - 3'd1] * w1[(k - 3'd1) * 6 + n];
            else             mac[n] = acc[n] + h[k - 3'd1] * ((n < 5) ? w2[(k - 3'd1) * 5 + n] : 16'sd0);
    end

    // argmax over the five logits, lowest index wins ties
    reg signed [31:0] best;
    reg [2:0] cls;
    always @* begin
        best = acc[0]; cls = 3'd0;
        for (n = 1; n < 5; n = n + 1)
            if (acc[n] > best) begin best = acc[n]; cls = n; end
    end

    always @(posedge clk) begin
        if (state == IDLE && start)
            for (n = 0; n < 7; n = n + 1) xr[n] <= x[16 * n +: 16];
        if (state == L1 && k == 3'd7)
            for (n = 0; n < 6; n = n + 1) begin
                shr = mac[n] >>> F1;
                h[n] <= (shr < 0) ? 16'sd0 : (shr > RELU6_MAX) ? RELU6_MAX[15:0] : shr[15:0];
            end
        if (state == L1 || state == L2)
            for (n = 0; n < 6; n = n + 1)
                acc[n] <= (k == 3'd0) ? ((state == L1) ? b1[n] : (n < 5 ? b2[n] : 32'sd0)) : mac[n];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; k <= 3'd0; valid <= 1'b0; class_out <= 3'd0;
        end else begin
            valid <= 1'b0;
            case (state)
                IDLE: if (start) begin state <= L1; k <= 3'd0; end
                L1:   if (k == 3'd7) begin state <= L2; k <= 3'd0; end else k <= k + 3'd1;
                L2:   if (k == 3'd6) begin state <= OUT; k <= 3'd0; end else k <= k + 3'd1;
                OUT:  begin class_out <= cls; valid <= 1'b1; state <= IDLE; end
            endcase
        end
    end
endmodule
