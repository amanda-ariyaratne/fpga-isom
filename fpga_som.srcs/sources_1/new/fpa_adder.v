`timescale 1ns / 1ps

module fpa_adder
(
    input wire clk,
    input wire [31:0] num1,
    input wire [31:0] num2,
    output wire [31:0] num_out,
    output wire is_done
);

reg [31:0] summation;
reg [2:0] bigger = 0; // 0 if equal, 1 if num1 bigger, 2 if num2 bigger
reg signed [7:0] exp_diff;
reg [31:0] n1;
reg [31:0] n2;
reg [23:0] m1;
reg [23:0] m2;
reg [7:0] e1;
reg [7:0] e2;
reg sign1;
reg sign2;

reg overflow;
reg normalize = 0;
reg init = 1;

reg [24:0] man_sum;
reg [5:0] shift_count=0;

reg done=0;

assign is_done=done;
assign num_out = summation;

always @(posedge clk) begin
    if (init) begin  
     
            // compare exponents
            if (num1[30:23] > num2[30:23])
                bigger = 1;
            else if (num2[30:23] > num1[30:23])
                bigger = 2;
            else begin
                if (num1[22:0] >= num2[22:0])
                    bigger = 1;
                else
                    bigger = 2;
            end
            
            // get numbers ordered
            if (bigger == 1) begin
                n1 = num1;
                n2 = num2;        
            end
            else begin
                n1 = num2;
                n2 = num1;  
            end
            
            e1 = n1[30:23];
            e2 = n2[30:23];
            
            m1 = {1'b1, n1[22:0]};
            m2 = {1'b1, n2[22:0]};
            
            // diff in exponents
            exp_diff = e1 - e2;
            
            // shift smaller one by exp difference
            m2 = m2 >> exp_diff;
            e2 = e1;
            
            sign1 = n1[31];
            sign2 = n2[31];
                
            
            // signs of the inputs are the same
            if (sign1 == sign2) begin
                man_sum = {1'b0, m1} + {1'b0, m2};  
                overflow = 0;  
                if (man_sum[24] == 1) begin
                    man_sum = man_sum >> 1;
                    overflow = 1;
                end
                    
                summation[31] = sign1;           
                summation[22: 0] =  man_sum[22: 0];
                summation[30: 23] = e1 + overflow;
            end

            else if (sign1 != sign2) begin
                $display("Diff");
                man_sum = m1 - m2;    
                summation[31] = sign1; 
            end
            init = 0;
            normalize = 1;

    end
end

always @(posedge clk) begin
    if (normalize) begin
        if (sign1 == sign2) begin
            done=1;
            normalize = 0;
        end
        else if (man_sum[23] == 0) begin
            shift_count = shift_count + 1;
            man_sum = man_sum << 1;
        end
        else begin
            summation[22: 0] =  man_sum[22: 0];
            summation[30: 23] = e1 - shift_count;
            done=1;
            normalize = 0;
        end        
    end
end

endmodule
