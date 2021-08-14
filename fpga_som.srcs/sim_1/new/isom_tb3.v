`timescale 1ns / 1ps

module isom_tb3();
    reg clk = 0;
    reg reset = 0;
    wire [8:0] prediction;
    
    isom uut(
        .clk(clk),
        .prediction(prediction)
    );
    
    reg [32:0] i=0;
    initial 
    begin
        for (i=0;i<100_000; i=i+1)
        begin
            clk = ~clk;
            #10;
        end
    end
    

endmodule
