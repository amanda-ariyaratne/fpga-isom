`timescale 1ns / 1ps

module isom_tb();
    reg clk = 0;
    reg reset = 0;
    wire [8:0] prediction;
    
//    isom uut(
//        .clk(clk),
//        .prediction(prediction)
//    );
    
//    reg [32:0] i=0;
//    initial 
//    begin
//        for (i=0;i<100_000; i=i+1)
//        begin
//            clk = ~clk;
//            #10;
//        end
//    end
    
    reg signed [1:0] a = 1;
    reg signed [1:0] b = -1;
    
    initial 
    begin
        prediction <= a*b;
    end

endmodule
