`timescale 1ns / 1ps
// 7-6-5 MLP with ReLU6, bit-exact with hw_fix/fixed_point_reference.py.
//
// Formats: x, h in Q4.12 (int16). W1 scaled 2^F1, W2 scaled 2^F2 (int16).
//          b1 scaled 2^(12+F1), b2 scaled 2^(12+F2) (int32). Accumulators int32.
// Layer 1: h[n] = clip((b1[n] + sum_k x[k]*W1[k][n]) >>> F1, 0, 6<<12)
// Layer 2: class = argmax_o (b2[o] + sum_k h[k]*W2[k][o]), lowest index wins ties.
// One multiply-accumulate per cycle: 6*8 + 5*7 + 2 = 85 cycles from start to valid.
module mlp_relu6 #(
    parameter F1 = 12,
    parameter F2 = 13,
    parameter W1_FILE = "w1.mem",   // 42 x int16, row-major [input][hidden]
    parameter B1_FILE = "b1.mem",   //  6 x int32
    parameter W2_FILE = "w2.mem",   // 30 x int16, row-major [hidden][output]
    parameter B2_FILE = "b2.mem"    //  5 x int32
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,      // one-cycle pulse; x is captured on this cycle
    input  wire [111:0] x,          // 7 x int16, feature k at x[16*k +: 16]
    output reg  [2:0]   class_out,  // 0 Walking, 1 Sitting, 2 Standing, 3 Laying, 4 Transitions
    output reg          valid,      // one-cycle pulse when class_out is ready
    output wire         busy
);
    localparam [15:0] RELU6_MAX = 16'd24576;  // 6.0 in Q4.12

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

    localparam IDLE = 2'd0, L1 = 2'd1, L2 = 2'd2;
    reg [1:0] state;
    reg [2:0] n;   // neuron being computed
    reg [2:0] k;   // 0 = load bias, 1..K = accumulate input k-1
    reg signed [31:0] acc, best;
    assign busy = (state != IDLE);

    // Current operands. Index math is small constants, so it stays cheap.
    wire signed [15:0] a_in  = (state == L1) ? xr[k - 3'd1] : h[k - 3'd1];
    wire signed [15:0] w_in  = (state == L1) ? w1[(k - 3'd1) * 6 + n] : w2[(k - 3'd1) * 5 + n];
    wire signed [31:0] mac   = acc + a_in * w_in;
    wire signed [31:0] shr   = mac >>> F1;   // used when the last layer-1 term lands
    wire        [15:0] relu6 = shr[31] ? 16'd0 : (shr > RELU6_MAX) ? RELU6_MAX : shr[15:0];

    // Datapath registers need no reset; keeping them out of the reset block avoids set/reset priority issues.
    integer i;
    always @(posedge clk) begin
        if (state == IDLE && start)
            for (i = 0; i < 7; i = i + 1) xr[i] <= x[16 * i +: 16];
        if (state == L1 && k == 3'd7)
            h[n] <= relu6;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; valid <= 1'b0; class_out <= 3'd0;
            n <= 3'd0; k <= 3'd0; acc <= 32'sd0; best <= 32'sd0;
        end else begin
            valid <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    state <= L1; n <= 3'd0; k <= 3'd0;
                end
                L1: begin
                    if (k == 3'd0) begin
                        acc <= b1[n]; k <= 3'd1;
                    end else if (k == 3'd7) begin          // last input term; h[n] latched above
                        k <= 3'd0;
                        if (n == 3'd5) begin state <= L2; n <= 3'd0; end
                        else n <= n + 3'd1;
                    end else begin
                        acc <= mac; k <= k + 3'd1;
                    end
                end
                L2: begin
                    if (k == 3'd0) begin
                        acc <= b2[n]; k <= 3'd1;
                    end else if (k == 3'd6) begin          // last hidden term
                        k <= 3'd0;
                        if (n == 3'd0 || mac > best) begin best <= mac; class_out <= n; end
                        if (n == 3'd4) begin state <= IDLE; valid <= 1'b1; end
                        else n <= n + 3'd1;
                    end else begin
                        acc <= mac; k <= k + 3'd1;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
