`timescale 1ns / 1ps

module fpa_tb();
    reg clk=0;
    reg [31:0] num1 = 32'b01000001000111000000000000000000;
    reg [31:0] num2 = 32'b00111111000100000000000000000000;
    wire [31:0] sum;
    wire done;
    
    fpa_adder uut(
        .clk(clk),
        .num1(num1),
        .num2(num2),
        .num_out(sum),
        .is_done(done)
    );
    
    integer i=0;
    initial 
    begin
        for (i=0;i<25; i=i+1)
        begin
            clk = ~clk;
            #10;
            if (done)
                $finish;
        end
    end
endmodule
