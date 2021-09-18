`timescale 1ns / 1ps

// only 2 to the power exponents

module to_ieee754
#(
    parameter DIGIT_DIM = 32
)
(
    input wire [DIGIT_DIM-1:0] number,
    input wire sign, // 0-> plus, 1-> minus
    input wire clk,
    input wire en,
    input wire reset,
    output wire is_done,
    output wire [DIGIT_DIM-1:0] out
);

reg [DIGIT_DIM-1:0] ieee_format;
reg done = 0;

assign out = ieee_format;
assign is_done = done;

always @(posedge clk) begin
    if (en) begin
        ieee_format[DIGIT_DIM-1] = sign;
        ieee_format[DIGIT_DIM-10: 0] = 0;
        ieee_format[DIGIT_DIM-2: DIGIT_DIM-9] = 127 + number;
        done = 1;
    end
end

always @(posedge clk) begin
    if (reset) begin
        done = 0;
    end
end

endmodule
