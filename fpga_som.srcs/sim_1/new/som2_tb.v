`timescale 1ns / 1ps

module som2_tb(

    );
    
    reg clk = 0;
    reg reset = 0;
    
    som uut(
        .clk(clk)
    );
    
    integer i=0;
    initial 
    begin
        for (i=0;i<100;i=i+1)
            begin
                clk = ~clk;
                #10;
            end
            
    end
endmodule