`timescale 1ns / 1ps


module som_tb2();

    reg clk = 0;
    integer i;
    
    som uut(.clk(clk));
    
    initial
    begin
        for (i = 0; i < 200; i=i+1)
            #10 clk = ~clk;
    end
endmodule
