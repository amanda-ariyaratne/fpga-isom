`timescale 1ns / 1ps

module fpa_tb();
    reg clk=0;
    reg reset=0;
    reg en=0;

    reg [32*4-1:0] num1 = 128'b00111111_10100000_00000000_00000000_01000000_00000000_00000000_00000000_01000000_00100000_00000000_00000000_01000000_01000000_00000000_00000000;
    reg [32*4-1:0] num2 = 128'b01000000_00000000_00000000_00000000_01000000_00000000_00000000_00000000_01000000_00000000_00000000_00000000_01000000_00000000_00000000_00000000;
    
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
        en=1;
        for (i=0;i<1000; i=i+1) begin
            clk = ~clk;
            #10;
            if (done) begin
                reset=1;
                count = count+1;
                if (count==2)
                    $finish;
                else begin                    
                    num1[32*1-1] = ~num1[32*1-1];
                end
            end
        end
    end
endmodule
