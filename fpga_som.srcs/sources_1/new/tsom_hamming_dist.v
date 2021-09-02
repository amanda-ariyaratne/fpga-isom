`timescale 1ns / 1ps

module tsom_hamming_dist 
#(
    parameter DIM = 1000,
    parameter LOG2_DIM = 10,
    parameter DIGIT_DIM = 2
)
(
    input clk,
    input en,
    input reset,
    input wire [(DIM*DIGIT_DIM)-1:0] w,
    input wire [(DIM*DIGIT_DIM)-1:0] x,
    output wire [LOG2_DIM:0] distance,
    output wire [LOG2_DIM:0] hash,
    output wire is_done
);

    reg [LOG2_DIM:0] hamming_dist = 0;
    reg [LOG2_DIM:0] hash_count = 0;
    reg [LOG2_DIM:0] i = 0;

    reg [DIGIT_DIM-1:0] w_i;
    reg [DIGIT_DIM-1:0] x_i;

    reg done = 0;

    assign distance = hamming_dist;
    assign hash = hash_count;
    assign is_done = done;

    always @(posedge reset) begin
        done = 0;
    end

    always @(posedge clk)
    begin 
        if (en) begin
            hamming_dist = 0;
            hash_count = 0;
            for (i = 0; i < DIM; i = i + 1) begin
                w_i = w[(i*DIGIT_DIM)+1 -:DIGIT_DIM];
                x_i = x[(i*DIGIT_DIM)+1 -:DIGIT_DIM];
                if ((w_i == 0 && x_i == 1) || (w_i == 1 && x_i == 0)) begin
                    hamming_dist = hamming_dist + 1;
                end else if (w_i == 2) begin
                    hash_count = hash_count + 1;
                end
            end
            done = 1;
        end
    end


endmodule