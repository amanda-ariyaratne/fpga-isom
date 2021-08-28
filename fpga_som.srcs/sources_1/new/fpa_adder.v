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
reg set_high_bit=0;
reg init=1;
reg done=0;

reg [23:0] man_sum;
reg [5:0] shift_count=0;

assign num_out = summation;
assign is_done = done;

always @(posedge clk) begin
    if (init) begin  
        m1[22:0] = num1[22:0];
        m1[23] = 1;
        m2[22:0] = num2[22:0];
        m2[23] = 1;
        e1 = num1[30:23] - 127;
        e2 = num2[30:23] - 127;
        
        // if one of the numbers are zero then answer is zero
        if ((e1 == 0) && (m1[22:0] == 0)) begin
            summation = 0;
            init = 0;
            done=1;
        end
        
        else if ((e2 == 0) && (m2[22:0] == 0)) begin
            summation = 0;
            init = 0;
            done=1;
        end
        
        else begin       
            // compare exponents
            if (e1 > e2)
                bigger = 1;
            else if (e2 > e1)
                bigger = 2;
            else if (e1 == e2) begin
                if (m1 >= m2)
                    bigger = 1;
                else
                    bigger = 2;
            end
            
            // get numbers ordered
            if (bigger == 1 || bigger ==0) begin
                n1 = num1;
                n2 = num2;        
            end
            else if (bigger == 2) begin
                n1 = num2;
                n2 = num1;  
            end
            
            m1[22:0] = n1[22:0];
            m2[22:0] = n2[22:0];
            m1[23] = 1;
            m2[23] = 1;
            e1 = n1[30:23] - 127;
            e2 = n2[30:23] - 127;
            sign1 = n1[31];
            sign2 = n2[31];
                
            // diff in exponents
            exp_diff = e1-e2;
            if (exp_diff > 8) begin
                summation = n1;
                init = 0;
                done=1;
            end
            
            // shift smaller one with number of exps diffs
            m2 = m2 >> exp_diff;
            
            // signes of the inputs are the same
            if (sign1 == sign2 && sign2 == 0) begin
                $display("Plus");
                {overflow, man_sum} = m1 + m2;    
                man_sum = man_sum >> overflow;
                if (overflow == 1)
                    man_sum[23] = overflow;
                    
                summation[31] = sign1;           
                summation[22: 0] =  man_sum[22: 0];
                summation[30: 23] = e1+overflow+127;
            end
            else if (sign1 == sign2 && sign2 == 1) begin
                $display("Minus");
                {overflow, man_sum} = m1 - m2;    
                man_sum = man_sum >> overflow;
                if (overflow == 1)
                    man_sum[23] = overflow;
                    
                summation[31] = sign1;           
                summation[22: 0] =  man_sum[22: 0];
                summation[30: 23] = e1+overflow+127;       
            end
            else if (sign1 != sign2) begin
                $display("Diff");
                man_sum = m1 - m2;    
                summation[31] = sign1; 
            end
            init = 0;
            set_high_bit = 1;
        
        end
    end
end

always @(posedge clk) begin
    if (set_high_bit) begin
        if (sign1 == sign2) begin
            set_high_bit=0;
            done=1;
        end
        if (man_sum[23] == 0) begin
            shift_count=shift_count+1;
            man_sum = man_sum << 1;
        end
        else begin
            summation[22: 0] =  man_sum[22: 0];
            summation[30: 23] = e1-shift_count+127;
            set_high_bit=0;
            done=1;
        end        
    end
end

endmodule
