`timescale 1ns / 1ps
// outputs maximum
// 0-> equal, 1->1, 2->2
module fpa_comparator
(
    input wire clk,
    input wire en,
    input wire reset,
    input wire [31:0] num1,
    input wire [31:0] num2,
    output wire [1:0] num_out,
    output wire is_done
);

reg [31:0] n1;
reg [31:0] n2;
reg [23:0] m1;
reg [23:0] m2;
reg signed [8:0] e1;
reg signed [8:0] e2;
reg sign1;
reg sign2;
reg [31:0] out;

reg [1:0] max = 0;
reg done=0;
reg init=1;

assign is_done = done;
assign num_out = max;

always @(posedge clk) begin
    if (en && init) begin
        m1[22:0] = num1[22:0];
        m2[22:0] = num2[22:0];
        m1[23] = 1;
        m2[23] = 1;
        e1 = num1[30:23] - 127;
        e2 = num2[30:23] - 127;
        sign1 = num1[31];
        sign2 = num2[31];
        
        if (sign1 != sign2) begin
            if (sign1 == 0) 
                max = 1;
            else
                max = 2;
        end
        
        if (sign1==sign2 && sign1==0) begin
            if (e1 > e2)
                max = 1;
            else if (e2 > e1)
                max = 2;
            else begin
                if (m1 > m2)
                    max = 1;
                else if (m2 > m1)
                    max = 2;
                else
                    max = 0;
            end
        end
        
        else if (sign1==sign2 && sign1==1) begin
            if (e1 < e2)
                max = 1;
            else if (e2 < e1)
                max = 2;
            else begin
                if (m1 < m2)
                    max = 1;
                else if (m2 < m1)
                    max = 2;
                else
                    max = 0;
            end
        end
        done = 1;
        init = 0;
    end
end

always @(posedge reset) begin
    done = 0;
    init = 1;
end

endmodule
