`timescale 1ns / 1ps
/**************************************
 LR(t +1) = alpha * psi(n) * LR(t)
 n = node count

 psi(n) = 1 -     R
              -----------
              2^(n/8 + 2)
***************************************/
module gsom_learning_rate
#(
    parameter R = 32'h40733333, // 3.8
    parameter R_EXP = 32'h3F800000, // 1
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

reg mul1_en = 0;
reg mul1_reset = 0;
reg [DIGIT_DIM-1:0] mul1_num1;
reg [DIGIT_DIM-1:0] mul1_num2;
wire [DIGIT_DIM-1:0] mul1_num_out;
wire mul1_is_done;    
fpa_multiplier multiplier1(
    .clk(clk),
    .en(mul1_en),
    .reset(mul1_reset),
    .num1(mul1_num1),
    .num2(mul1_num2),
    .num_out(mul1_num_out),
    .is_done(mul1_is_done)
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

reg add1_en = 0;
reg add1_reset = 0;
reg [DIGIT_DIM-1:0] add1_num1;
reg [DIGIT_DIM-1:0] add1_num2;
wire [DIGIT_DIM-1:0] add1_num_out;
wire add1_is_done; 
fpa_adder adder1(
    .clk(clk),
    .en(add1_en),
    .reset(add1_reset),
    .num1(add1_num1),
    .num2(add1_num2),
    .num_out(add1_num_out),
    .is_done(add1_is_done)
);

reg [DIGIT_DIM-1:0] out;
reg done = 0;
reg init = 1;

reg step_1 = 0;
reg step_2 = 0;
reg step_3 = 0;
reg step_4 = 0;

reg signed [23:0] shift_count;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        done = 0;
        init = 1;
        
    end else if (en && init) begin
        /*************
        2^(n/8 + 2)
        **************/  
        out = 0;
        
        add_num1 = 32'h40000000; // 2
        add_num2[31] = 0;
        add_num2[30:23] = node_count[30:23] - 3;
        add_num2[22:0] = node_count[22:0];
        
        add_en = 1;
        add_reset = 0;
        
        // prev lr * alpha
        mul_num1 = prev_learning_rate;
        mul_num2 = alpha;
        
        mul_en = 1;
        mul_reset = 0;
        
        init = 0; 
        step_4 = 1;  
//        step_1 = 1;  
        
    end else if (en && step_1 && add_is_done) begin
        
        /**************************************
              R
          -----------
          2^(n/8 + 2)  
        ***************************************/
//        $display("1 %h", add_num_out);
        add_en = 0;
        add_reset = 1;
                
        add1_num1 = R;
        add1_num2 = add_num_out;
        add1_num2[31] = 1;
        
        add1_en = 1;
        add1_reset = 0;
        
        step_1 = 0; 
        step_2 = 1;   
           
    end else if (step_2 && add1_is_done) begin
//        $display("2 %h", add1_num_out);
        add1_en = 0;
        add1_reset = 1;   
             
        add_num1 = 32'h3F800000;    // 1
        add_num2[31] = 1;
        add_num2[22:0] = R[22:0];
        
        shift_count = add1_num_out[30:23]-127;
        if (shift_count > 0)
            add_num2[30:23] = add1_num_out[22:0] >> (23 - shift_count);    
        else
            add_num2[30:23] = 0;
        add_num2[30:23] = add_num2[30:23] + 127;

        add_en = 1;
        add_reset = 0;
        
        step_2 = 0;
        step_3 = 1;
        
    end else if (step_3 && add_is_done && mul_is_done) begin
//        $display("3 %h", mul_num_out);
//        $display("4 %h", add_num_out);
        
        add_en = 0;
        add_reset = 1;        
        
        mul_en = 0;
        mul_reset = 1;
        
        mul1_num1 = mul_num_out;
        mul1_num2 = add_num_out;
        mul1_en = 1;
        mul1_reset = 0;
        
        step_3 = 0;
        step_4 = 1;
    end else if (step_4 && mul_is_done) begin
//        $display("4 %h", mul_num_out);
        mul_en = 0;
        mul_reset = 1;
        
        done = 1;
        out = mul_num_out;
        
        step_4 = 0;
    end
end

assign is_done = done;
assign learning_rate = out;

endmodule
