`timescale 1ns / 1ps

module fpa_distance
(
    input wire clk,
    input wire reset,
    input wire [31:0] num1,
    input wire [31:0] num2,
    output wire [31:0] num_out,
    output wire is_done
);

reg [31:0] subtraction;
reg [31:0] square;

wire [31:0] subtraction_out;
wire [31:0] square_out;

reg [31:0] sub_in_1;
reg [31:0] sub_in_2;
reg [31:0] square_in;

reg done=0;
wire subtraction_done;
wire square_done;

assign is_done=done;
assign num_out = square;

fp_adder subtraction_unit(
    .clk(clk),
    .num1(sub_in_1),
    .num2(sub_in_2),
    .num_out(subtraction_out),
    .is_done(subtraction_done)
);

fp_multiplier square_unit(
    .clk(clk),
    .num1(square_in),
    .num2(square_in),
    .num_out(square_out),
    .is_done(square_done)
);

always @(posedge clk) begin   
    sub_in_1 = num1;
    sub_in_2 = num2;
    square_in = subtraction_out;
    square = square_out;
    
    if (square_done) begin
        done = 1;
    end
end

always @(posedge reset) begin
    done = 0;
end

endmodule
