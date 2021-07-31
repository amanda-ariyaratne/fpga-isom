`timescale 1ns / 1ps

module som_sim_tb();
    localparam TEST_ROWS = 2;
    reg clk = 0;
    reg reset = 0;
    wire [TEST_ROWS-1:0] prediction;
    
    som uut(
        .clk(clk),
        .prediction(prediction)
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
