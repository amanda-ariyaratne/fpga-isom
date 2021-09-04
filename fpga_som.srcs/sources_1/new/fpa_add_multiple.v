`timescale 1ns / 1ps

module fpa_add_multiple
#(
    parameter DIM=4,
    parameter DIGIT_DIM=32,
    parameter ELEMENT_COUNT=4
)
(
    input wire clk,
    input wire en,
    input wire reset,
    input wire [DIGIT_DIM*ELEMENT_COUNT-1:0] num_array,
    output wire [DIGIT_DIM-1:0] num_out,
    output wire is_done
);

reg done = 0;
reg [DIGIT_DIM-1:0] out;
reg init = 1; 

integer signed i = DIGIT_DIM;

//////////////////adder unit//////////////////////
reg adder_en;
reg adder_reset;
reg [DIGIT_DIM-1:0] adder_in_1 = 0;
reg [DIGIT_DIM-1:0] adder_in_2;
wire [DIGIT_DIM-1:0] adder_out;
wire adder_done;

fpa_adder adder(
    .clk(clk),
    .en(adder_en),
    .reset(adder_reset),
    .num1(adder_in_1),
    .num2(adder_in_2),
    .num_out(adder_out),
    .is_done(adder_done)
);


always @(posedge clk) begin 
    if (en && init) begin
        adder_in_2 = num_array[i-1 -:DIGIT_DIM];
        adder_reset = 0;
        adder_en = 1;
        init = 0;
    end
end

always @(posedge clk) begin 
    if (adder_done) begin
        adder_en = 0;
        adder_reset = 1;
        adder_in_1 = adder_out;
        
        i = i + DIGIT_DIM;
        if (i < DIGIT_DIM * ELEMENT_COUNT) begin
            init = 1;
        end else begin
            done = 1;
            out = adder_out;
        end
    end
end

////////////////////////////////////////


always @(posedge reset) begin
    done=0;
    init=1;
end

assign num_out = out;
assign is_done = done;


endmodule
