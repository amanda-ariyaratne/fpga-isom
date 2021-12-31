`timescale 1ns / 1ps

module fpa_update_weight

(
    input wire clk,
    input wire en,
    input wire reset,
    input wire [31:0] weight,
    input wire [31:0] train_row,
    input wire [31:0] alpha,
    output wire [31:0] num_out,
    output wire is_done    
);

reg [31:0] out;

reg en_1=1;
reg en_2=0;
reg en_3=0;
reg en_4=0;

reg done=0;

reg [31:0] add_in_1;
reg [31:0] add_in_2;
wire [31:0] add_out;
wire add_done;

reg [31:0] mul_in_1;
reg [31:0] mul_in_2;
wire [31:0] mul_out;
reg mul_reset;
wire mul_done;

reg en_add=0;
reg en_mul=0;
reg add_reset;

assign num_out = add_out;
assign is_done = done;

fpa_adder add(
    .clk(clk),
    .en(en_add),
    .reset(add_reset),
    .num1(add_in_1),
    .num2(add_in_2),
    .num_out(add_out),
    .is_done(add_done)
);

fpa_multiplier multiply(
    .clk(clk),
    .en(en_mul),
    .reset(mul_reset),
    .num1(mul_in_1),
    .num2(mul_in_2),
    .num_out(mul_out),
    .is_done(mul_done)
);

always @(posedge clk or posedge reset) begin
    if (reset) begin
        done = 0;
        en_1=1;
    end
    else if (en && en_1) begin
        add_reset = 0;
        add_in_1 = weight;
        add_in_2 = train_row;
        add_in_1[31] = 1; // indicate subtraction
        en_add = 1; // on the adder module
        
        en_1=0; // off this block
        en_2=1; // on next block
    end else if (en && en_2 && add_done) begin
        en_add=0; // off adder module
        add_reset = 1;
        
        mul_reset = 0;
        mul_in_1 = add_out;
        mul_in_2 = alpha;
        en_mul = 1; // on the adder module
        
        en_2=0; // off this block
        en_3=1; // on next block
    end else if (en && en_3 && mul_done) begin
        en_mul=0; // off multi module
        mul_reset = 1;
        
        add_reset = 0;
        add_in_1 = weight;
        add_in_2 = mul_out;
        en_add = 1; // on the adder module
        
        en_3=0; // off this block
        en_4=1; // on next block
    end else if (en && en_4 && add_done) begin
        en_add = 0;
        add_reset=1;
        done=1;
        en_4=0;
    end
end

endmodule
