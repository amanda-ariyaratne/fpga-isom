`timescale 1ns / 1ps

module gsom_error_update
#(
    parameter DIM=4,
    parameter DIGIT_DIM=32
)
(
    input wire clk,
    input wire en,
    input wire reset,
    input wire [31:0] error,
    input wire [31:0] fd_ratio,
    output wire [31:0] updated_error,
    output wire is_done    
);

reg [31:0] out;
reg init=1;
reg done=0;

reg [31:0] mul_in_1;
reg [31:0] mul_in_2;
wire [31:0] mul_out;
reg mul_reset;
wire [1:0] mul_done;

reg mul_en=0;

assign updated_error = mul_out;
assign is_done = done;

fpa_multiplier multiply(
    .clk(clk),
    .en(en_mul),
    .reset(mul_reset),
    .num1(mul_in_1),
    .num2(mul_in_2),
    .num_out(mul_out),
    .is_done(mul_done)
);

always @(posedge reset) begin
    done = 0;
    init=1;
end

always @(posedge clk) begin
    if (en && init) begin
        mul_in_1 = error;
        mul_in_2 = fd_ratio;
        mul_reset = 0;
        mul_en = 1;
        
        init=0;
    end
    if (mul_done) begin
        mul_reset = 1;
        mul_en = 0;
        done = 1;
    end
end

endmodule