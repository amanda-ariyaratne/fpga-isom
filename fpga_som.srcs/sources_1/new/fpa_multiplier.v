`timescale 1ns / 1ps

module fpa_multiplier(
    input wire clk,
    input wire en,
    input wire reset,
    input wire[31:0] num1,
    input wire[31:0] num2,
    output wire[31:0] num_out,
    output wire is_done
);

wire sgn1;
wire sgn2;
reg sgn_out;

reg [7:0] exp1;
reg [7:0] exp2;
reg [7:0] exp_out;

reg [23:0] man1;
reg [23:0] man2;
reg [23:0] man_out;

reg [47:0] product;
reg done=0;
reg init=1;

assign sgn1 = num1[31];
assign sgn2 = num2[31];
assign num_out = {sgn_out, exp_out, man_out[22:0]};
assign is_done = done;

always @(posedge clk or posedge reset)
begin
    if (reset) begin  
        done = 0;
        init = 1;
    end else begin
    
        if (en && init) begin
            if ((num1[30:23] == 0 && num1[22:0] == 0) || (num2[30:23] == 0 && num2[22:0] == 0)) begin
                sgn_out = 0;
                exp_out = 0;
                man_out = 0;
            
            end else if ((num1[30:23] == 8'b11111111 && num1[22:0] == 0) || (num2[30:23] == 8'b11111111 && num2[22:0] == 0)) begin
                sgn_out = sgn1 ^ sgn2;
                exp_out = 8'b11111111;
                man_out = 0;
        
            end else begin
                sgn_out = sgn1 ^ sgn2;
                
                if(num1[30:23] == 0) begin
                    exp1 = 8'b00000001;
                    man1 = {1'b0, num1[22:0]};
                end else begin
                    exp1 = num1[30:23];
                    man1 = {1'b1, num1[22:0]};
                end
                
                if(num2[30:23] == 0) begin
                    exp2 = 8'b00000001;
                    man2 = {1'b0, man2[22:0]};
                end else begin
                    exp2 = num2[30:23];
                    man2 = {1'b1, num2[22:0]};
                end
            
                exp_out = exp1 + exp2 - 127;
                product = man1 * man2;
        
                if (product[47] == 1) begin
                    exp_out = exp_out + 1;
                end else begin
                    product = product << 1;
                end
        
                man_out = product[47:24];
        
                if (exp_out < 1) begin
                    sgn_out = 1;
                    exp_out = 8'b11111111;
                    man_out = 0;
                end else if (exp_out > 254) begin
                    sgn_out = 1;
                    exp_out = 8'b11111111;
                    man_out = 0;
                end
            end
            done = 1;
            init = 0;
        end
    end 
    
end



endmodule