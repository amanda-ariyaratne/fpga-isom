`timescale 1ns / 1ps

module gsom_tb();
    reg clk = 0;
    wire [31:0] out;
    wire completed;
    reg reset = 1;
    reg en = 0;
    
    gsom uut(
        .clk(clk)
    );
    
//    gsom_learning_rate uut(
//        .clk(clk),
//        .en(en),
//        .reset(reset),
//        .node_count(32'h40800000),
//        .prev_learning_rate(32'h3F666666),
//        .alpha(32'h3F000000),
//        .learning_rate(out),
//        .is_done(completed)                
//    );
    
    reg [32:0] i=0;
    initial begin
//        reset = 0;
//        en = 1;
        for (i=0;i<1000_000; i=i+1) begin
            clk = ~clk;
            if (completed)
                $finish;
            #10;
        end
        $finish;
    end
    

endmodule

