`timescale 1ns / 1ps

module fpa_tb();
    reg clk=1;
    
    reg reset;
    reg en;
    reg [32*4-1:0] num1 = 128'h3e471c713f2000003d50456c3daaaaaa;
    reg [32*4-1:0] num2 = 128'h3f33d9973e2b549b3f3f861b3ea541e8;
    reg [32*4-1:0] alpha = 32'b00111111_00000000_00000000_00000000;
    wire [31:0] out;
    wire done;
        
    fpa_euclidean_distance uut(
        .clk(clk),
        .reset(reset),
        .en(en),
        .weight(num1),
        .trainX(num2),
        .num_out(out),
        .is_done(done)
    );
    
    integer i=0;
    integer count=0;
    initial begin        
        for (i=0;i<1000; i=i+1) begin
            clk = ~clk;
            #10;
            reset=0;
            en=1;
            
            if (done) begin
                reset=1; 
                en=0;
                count = count+1;
                if (count==4)
                    $finish;
                else begin
                    $display("num2 ", num2);
                    num2[32*1-1] = ~num2[32*1-1];
                end
            end
        end
    end
endmodule
