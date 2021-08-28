`timescale 1ns / 1ps

module fpa_update_weight
(
    input wire clk,
    input wire reset,
    input wire [31:0] weight,
    input wire [31:0] train_row,
    input wire [31:0] alpha,
    output wire [1:0] num_out,
    output wire is_done    
);

reg done=0;

reg [31:0] add_in_1;
reg [31:0] add_in_2;
wire [31:0] add_out;
reg add_reset;
wire [1:0] add_done;

reg [31:0] mul_in_1;
reg [31:0] mul_in_2;
wire [31:0] mul_out;
reg mul_reset;
wire [1:0] mul_done;

fpa_adder add(
    .clk(clk),
    .reset(add_reset),
    .num1(add_in_1),
    .num2(add_in_2),
    .num_out(add_out),
    .is_done(add_done)
);

fpa_multiplier multiply(
    .clk(clk),
    .reset(mul_reset),
    .num1(mul_in_1),
    .num2(mul_in_2),
    .num_out(mul_out),
    .is_done(mul_done)
);

reg [31:0] out;

always @(posedge clk) begin
    add_in_1 = weight;
    add_in_2 = train_row;
    add_in_2[31] = 1; // indicate subtraction
    
    if (add_done) begin
        mul_in_1 = add_out;
        mul_in_2 = alpha;
        
        if (mul_done) begin
            
        end
    end
    
end

always @(posedge reset) begin
    done = 0;
end

endmodule
