`timescale 1ns / 1ps


// LR(t +1) = alpha * psi(n) * LR(t)
// n = node count

// psi(n) = 1 -     R
//              -----------
//              2^(n/8 + 2)

module gsom_learning_rate
#(
    parameter R = 32'h40733333,
    parameter DIGIT_DIM = 32
)
(
    input wire clk,
    input wire en,
    input wire reset,
    input wire [DIGIT_DIM-1:0] node_count,
    input wire [DIGIT_DIM-1:0] prev_learning_rate,
    input wire [DIGIT_DIM-1:0] alpha,
    output wire [DIGIT_DIM-1:0] learning_rate,
    output wire is_done
);

reg mul_en = 0;
reg mul_reset = 0;
reg [DIGIT_DIM-1:0] mul_num1;
reg [DIGIT_DIM-1:0] mul_num2;
wire [DIGIT_DIM-1:0] mul_num_out;
wire mul_is_done;    
fpa_multiplier multiplier(
    .clk(clk),
    .en(mul_en),
    .reset(mul_reset),
    .num1(mul_num1),
    .num2(mul_num2),
    .num_out(mul_num_out),
    .is_done(mul_is_done)
);

reg add_en = 0;
reg add_reset = 0;
reg [DIGIT_DIM-1:0] add_num1;
reg [DIGIT_DIM-1:0] add_num2;
wire [DIGIT_DIM-1:0] add_num_out;
wire add_is_done; 
fpa_adder adder(
    .clk(clk),
    .en(add_en),
    .reset(add_reset),
    .num1(add_num1),
    .num2(add_num2),
    .num_out(add_num_out),
    .is_done(add_is_done)
);

reg done = 0;
reg init = 1;
always @(posedge reset) begin
    done = 0;
    init = 1;
end

reg [DIGIT_DIM-1:0] out;
reg mul_1_en = 0;
reg mul_2_en = 0;
reg mul_3_en = 0;
always @(posedge reset) begin
    if (en && init) begin
        
        add_num1 = 32'h40000000; // 2
        ///****** 2^(x-3) X 1.y ******///
        add_num2[31] = 0; // sign bit
        add_num2[30:23] = node_count[30:23] - 3; // exponent - 3
        add_num2[22:0] = node_count[22:0]; // mantissa is the same
        
        add_en = 1;
        add_reset = 0;
        init = 0; 
        mul_1_en = 1;       
    end
    
    if (add_is_done && mul_1_en) begin
        add_en = 0;
        add_reset = 1;
        
        add_num1 = 32'h3F800000;
        add_num2[31] = 1; // indicate subtraction
        add_num2[30:23] = R[30:23] - add_num_out;
        add_num2[22:0] = R[22:0];
        add_en = 1;
        add_reset = 0;
        
        // prev lr * alpha
        mul_num1 = prev_learning_rate;
        mul_num2 = alpha;
        mul_en = 1;
        mul_reset = 0;
        
        mul_2_en = 1;
        mul_1_en = 0;
    end
    
    if (add_is_done && mul_is_done && mul_2_en) begin
        add_en = 0;
        add_reset = 1;        
        
        mul_en = 0;
        mul_reset = 1;
        
        mul_num1 = mul_num_out;
        mul_num2 = add_num_out;
        mul_en = 1;
        mul_reset = 0;
        
        mul_2_en = 0;
        mul_3_en = 1;
    end
    
    if (mul_is_done && mul_3_en) begin
        mul_en = 0;
        mul_reset = 1;
        
        done = 1;
        out = mul_num_out;
        
        mul_3_en = 0;
    end
end

assign is_done = done;
assign learning_rate = out;

endmodule
