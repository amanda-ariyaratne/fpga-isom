`timescale 1ns / 1ps


module fp_adder_tb();
    reg clk=0;
    reg [31:0] num1 = 32'b01110010111111100001000000000000;
    reg [31:0] num2 = 32'b01110010111111101001000000000000;
    wire [31:0] sum;
    
    fpa_comparator uut(
        .clk(clk),
        .num1(num1),
        .num2(num2),
        .num_out(sum)
    );
    
    integer i=0;
    initial 
    begin
        for (i=0;i<25; i=i+1)
        begin
            clk = ~clk;
            #10;
        end
    end
endmodule
