`timescale 1ns / 1ps

module fpa_tb();
    reg clk=0;
    reg reset=0;
    reg en=0;

    reg [32*4-1:0] num1 = 32'b00111111_10100000_00000000_00000000;
    reg [32*4-1:0] num2 = 32'b01000000_00000000_00000000_00000000;
    
    reg [32*4-1:0] alpha = 32'b00111111_00000000_00000000_00000000;
    
    wire [31:0] out;
    wire done;
    
    fpa_update_weight uut(
        .clk(clk),
        .reset(reset),
        .en(en),
        .weight(num1),
        .train_row(num2),
        .alpha(alpha),
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
                if (count==1)
                    $finish;
                else begin                    
                    num1[32*1-1] = ~num1[32*1-1];
                end
            end
        end
    end
endmodule
