`timescale 1ns / 1ps

module tsom_tb();
    reg clk = 0;
    reg reset = 0;
    wire [8:0] prediction;
    wire completed;
    
    tsom uut(
        .clk(clk),
        .prediction(prediction),
        .completed(completed)
    );
    
    reg [32:0] i=0;
    initial 
    begin
        for (i=0;i<1000_000; i=i+1)
        begin
            clk = ~clk;
            #10;
            if (completed)
                $finish;
        end
    end
    
endmodule