`timescale 1ns / 1ps

module fpa_distance
#(
    parameter DIM=4,
    parameter DIGIT_DIM=32
)
(
    input wire clk,
    input wire en,
    input wire reset,
    input wire [31:0] num1,
    input wire [31:0] num2,
    output wire [31:0] num_out,
    output wire is_done
);

reg init=1;
reg done=0;

reg sub_en=0;
reg sub_reset=0;
reg [31:0] sub_in_1;
reg [31:0] sub_in_2;
wire [31:0] subtraction_out;
wire subtraction_done;

reg squrae_en=0;
reg squrae_reset=0;
reg [31:0] square_in;
wire [31:0] square_out;
wire square_done;

fpa_adder subtraction_unit(
    .clk(clk),
    .en(sub_en),
    .reset(sub_reset),
    .num1(sub_in_1),
    .num2(sub_in_2),
    .num_out(subtraction_out),
    .is_done(subtraction_done)
);

fpa_multiplier square_unit(
    .clk(clk),
    .en(squrae_en),
    .reset(squrae_reset),
    .num1(square_in),
    .num2(square_in),
    .num_out(square_out),
    .is_done(square_done)
);

always @(posedge clk or posedge reset) begin 
    if (reset) begin
        done = 0;
        init=1;
    end
    else if (en && init) begin  
        sub_in_1 = num1;
        sub_in_2 = num2;
        sub_in_2[DIGIT_DIM-1] = ~sub_in_2[DIGIT_DIM-1]; // flip the sign bit
        sub_en=1;
        sub_reset=0;
        init=0;
    end
    
    else if (subtraction_done && !squrae_en) begin
        sub_en=0;
        sub_reset=1;
        
        square_in = subtraction_out;
        squrae_en=1;
        squrae_reset=0;
    end
    
    else if (square_done) begin 
        done = 1;    
        squrae_en=0;        
        squrae_reset=1; 
    end
end

assign is_done = done;
assign num_out = square_out;

endmodule
