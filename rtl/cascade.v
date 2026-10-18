`timescale 1ns / 1ps
// Two-stage early-exit HAR classifier, bit-exact with hw_fix/cascade_hw.py.
//
// Per window: stage 1 (7-6-5 ReLU6) on the accelerometer features, then the gate.
//   margin  = top1 - top2 stage-1 logit;  c1 = stage-1 class
//   trigger = MASK[c1] && margin < TAU                    -> switch the second sensor on
//   in_set  = MASK[c1] && (KEEP_ALL || margin < TAU_KEEP) -> keep it on
//   covered = sensor on for two whole windows already (switched on at end of w, first covers w+2)
//   covered && in_set -> stage 2 (N2-H2-5 ReLU6) on [x, g], argmax over MASK2; else answer with c1.
// One multiply-accumulate engine for all four layers. Weights are stored neuron-major in MAC order,
// so one pointer walks the weight ROM and one walks the bias ROM.
// Feature loading takes N2 cycles. Cycles start->valid: stage 1 = 6*8 + 5*7 + 1 = 84; stage 2 adds H2*(N2+1) + 5*(H2+1).
module cascade #(
    parameter N2 = 9,               // stage-2 inputs: 7 accelerometer features + N2-7 second-sensor features
    parameter H2 = 8,               // stage-2 hidden neurons
    parameter FA = 12,              // weight fraction bits, stage-1 hidden layer
    parameter FB = 12,              // weight fraction bits, stage-2 hidden layer
    parameter signed [32:0] TAU = 0,
    parameter signed [32:0] TAU_KEEP = 0,
    parameter KEEP_ALL = 1,
    parameter [4:0] MASK  = 5'b00110,   // bit c = class c may use the second sensor
    parameter [4:0] MASK2 = 5'b00110,   // classes stage 2 may answer
    parameter W_FILE = "w.mem",     // int16: s1 W1, s1 W2, s2 W1, s2 W2, each neuron-major
    parameter B_FILE = "b.mem"      // int32: s1 b1, s1 b2, s2 b1, s2 b2
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire [15:0]     x_in,        // features streamed in order 0..N2-1, Q4.12 (7 accel, then N2-7 second sensor)
    input  wire            x_valid,     // one feature per cycle while idle
    input  wire            start,       // one-cycle pulse after the N2 features
    output reg  [2:0]      class_out,
    output reg             valid,
    output wire            busy,
    output reg             sensor_en,   // power enable for the second sensor, updated with valid
    output reg             used_s2      // this answer came from stage 2
);
    localparam NW = 7 * 6 + 6 * 5 + N2 * H2 + H2 * 5;
    localparam NB = 6 + 5 + H2 + 5;
    localparam HN = H2 > 6 ? H2 : 6;
    localparam [15:0] RELU6_MAX = 16'd24576;

    reg signed [15:0] w [0:NW-1];
    reg signed [31:0] b [0:NB-1];
    initial begin
        $readmemh(W_FILE, w);
        $readmemh(B_FILE, b);
    end

    reg signed [15:0] xr [0:N2-1];
    reg signed [15:0] h  [0:HN-1];

    localparam IDLE = 3'd0, L0 = 3'd1, L1 = 3'd2, GATE = 3'd3, L2 = 3'd4, L3 = 3'd5;
    reg [2:0] state;
    reg [4:0] n, k;                      // neuron; k = 0 load bias, 1..n_in accumulate input k-1
    reg [$clog2(NW)-1:0] wa;
    reg [$clog2(NB)-1:0] ba;
    reg signed [31:0] acc, best, second;
    reg [2:0] cls;
    reg found;
    reg [1:0] cnt;                       // whole windows the sensor has been on, saturates at 2
    assign busy = (state != IDLE);

    wire first   = (state == L0 || state == L2);   // hidden layers read x, output layers read h
    wire [4:0] n_in   = (state == L0) ? 5'd7 : (state == L1) ? 5'd6 : (state == L2) ? N2 : H2;
    wire [4:0] n_last = (state == L0) ? 5'd5 : (state == L2) ? H2 - 1 : 5'd4;
    wire signed [15:0] a_in  = first ? xr[k - 5'd1] : h[k - 5'd1];
    wire signed [31:0] mac   = acc + a_in * w[wa];
    wire signed [31:0] shr   = (state == L0) ? (mac >>> FA) : (mac >>> FB);
    wire        [15:0] relu6 = shr[31] ? 16'd0 : (shr > RELU6_MAX) ? RELU6_MAX : shr[15:0];
    wire last_term = (k == n_in);

    wire signed [32:0] margin  = {best[31], best} - {second[31], second};
    wire in_set  = MASK[cls] && (KEEP_ALL || margin < TAU_KEEP);
    wire trigger = MASK[cls] && margin < TAU;
    wire take    = MASK2[n] && (!found || mac > best);   // stage-2 masked argmax, lowest index wins ties

    integer i;
    always @(posedge clk) begin
        if (state == IDLE && x_valid) begin  // shift register: after N2 writes xr[k] = feature k
            for (i = 0; i < N2 - 1; i = i + 1) xr[i] <= xr[i + 1];
            xr[N2 - 1] <= x_in;
        end
        if (first && last_term)
            h[n] <= relu6;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; valid <= 1'b0; class_out <= 3'd0; sensor_en <= 1'b0; used_s2 <= 1'b0;
            n <= 5'd0; k <= 5'd0; wa <= 0; ba <= 0; cnt <= 2'd0;
            acc <= 32'sd0; best <= 32'sd0; second <= 32'sd0; cls <= 3'd0; found <= 1'b0;
        end else begin
            valid <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    state <= L0; n <= 5'd0; k <= 5'd0; wa <= 0; ba <= 0;
                end
                GATE: begin
                    if (sensor_en && cnt == 2'd2 && in_set) begin
                        state <= L2; found <= 1'b0; used_s2 <= 1'b1;
                    end else begin
                        state <= IDLE; valid <= 1'b1; class_out <= cls; used_s2 <= 1'b0;
                    end
                    if (sensor_en) begin
                        if (in_set) cnt <= (cnt == 2'd2) ? 2'd2 : cnt + 2'd1;
                        else begin sensor_en <= 1'b0; cnt <= 2'd0; end
                    end else if (trigger) begin
                        sensor_en <= 1'b1; cnt <= 2'd1;
                    end
                end
                L0, L1, L2, L3: begin
                    if (k == 5'd0) begin
                        acc <= b[ba]; ba <= ba + 1'b1; k <= 5'd1;
                    end else if (!last_term) begin
                        acc <= mac; wa <= wa + 1'b1; k <= k + 5'd1;
                    end else begin                      // last term of neuron n; h[n] latched above
                        wa <= wa + 1'b1; k <= 5'd0;
                        if (state == L1) begin          // track top two logits for the margin
                            if (n == 5'd0 || mac > best) begin
                                second <= (n == 5'd0) ? 32'sh80000000 : best; best <= mac; cls <= n[2:0];
                            end else if (mac > second) second <= mac;
                        end
                        if (state == L3 && take) begin best <= mac; cls <= n[2:0]; found <= 1'b1; end
                        if (n == n_last) begin
                            n <= 5'd0;
                            case (state)
                                L0: state <= L1;
                                L1: state <= GATE;
                                L2: state <= L3;
                                default: begin state <= IDLE; valid <= 1'b1; class_out <= take ? n[2:0] : cls; end
                            endcase
                        end else n <= n + 5'd1;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
