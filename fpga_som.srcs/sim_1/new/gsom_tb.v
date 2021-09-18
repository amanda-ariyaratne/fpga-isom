`timescale 1ns / 1ps

module gsom_tb();
    reg clk = 0;
//    reg reset = 0;
//    wire [8:0] prediction;
//    wire completed;
    
    gsom uut(
        .clk(clk)
    );
    
    reg [32:0] i=0;
    initial begin
        for (i=0;i<1000_000_000; i=i+1) begin
            clk = ~clk;
            #10;
        end
    end
    

endmodule

