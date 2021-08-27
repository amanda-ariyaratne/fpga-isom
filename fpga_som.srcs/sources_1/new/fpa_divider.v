`timescale 1ns / 1ps

module fpa_divider(
    input wire clk,
    input wire[31:0] num1,
    input wire[31:0] num2,
    output wire[31:0] num_out
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

    reg [23:0] division;

    assign sgn1 = num1[31];
    assign sgn2 = num2[31];
    assign num_out = {sgn_out, exp_out, man_out[22:0]};

    always @(posedge clk)
    begin

        if (num1[30:23] == 0 && num1[22:0] == 0) begin
            if (!(num2[30:23] == 0 && num2[22:0] == 0)) begin
                sgn_out = 0;
                exp_out = 0;
                man_out = 0;
            end else begin
                sgn_out = 0;
                exp_out = 8'b111111;
                man_out = 23'b10000000000000000000000;
            end
        end else if (num2[30:23] == 0 && num2[22:0] == 0) begin
            sgn_out = 0;
            exp_out = 8'b111111;
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

            exp_out = exp1 - exp2 + 127;
            division = man1 / man2;

            if (division[23] == 1) begin
                division = division << 0;
                exp_out = exp_out - 0;
            end else if (division[22] == 1) begin
                division = division << 1;
                exp_out = exp_out - 1;
            end else if (division[21] == 1) begin
                division = division << 2;
                exp_out = exp_out - 2;
            end else if (division[20] == 1) begin
                division = division << 3;
                exp_out = exp_out - 3;
            end else if (division[19] == 1) begin
                division = division << 4;
                exp_out = exp_out - 4;
            end else if (division[18] == 1) begin
                division = division << 5;
                exp_out = exp_out - 5;
            end else if (division[17] == 1) begin
                division = division << 6;
                exp_out = exp_out - 6;
            end else if (division[16] == 1) begin
                division = division << 7;
                exp_out = exp_out - 7;
            end else if (division[15] == 1) begin
                division = division << 8;
                exp_out = exp_out - 8;
            end else if (division[14] == 1) begin
                division = division << 9;
                exp_out = exp_out - 9;
            end else if (division[13] == 1) begin
                division = division << 10;
                exp_out = exp_out - 10;
            end else if (division[12] == 1) begin
                division = division << 11;
                exp_out = exp_out - 11;
            end else if (division[11] == 1) begin
                division = division << 12;
                exp_out = exp_out - 12;
            end else if (division[10] == 1) begin
                division = division << 13;
                exp_out = exp_out - 13;
            end else if (division[9] == 1) begin
                division = division << 14;
                exp_out = exp_out - 14;
            end else if (division[8] == 1) begin
                division = division << 15;
                exp_out = exp_out - 15;
            end else if (division[7] == 1) begin
                division = division << 16;
                exp_out = exp_out - 16;
            end else if (division[6] == 1) begin
                division = division << 17;
                exp_out = exp_out - 17;
            end else if (division[5] == 1) begin
                division = division << 18;
                exp_out = exp_out - 18;
            end else if (division[4] == 1) begin
                division = division << 19;
                exp_out = exp_out - 19;
            end else if (division[3] == 1) begin
                division = division << 20;
                exp_out = exp_out - 20;
            end else if (division[2] == 1) begin
                division = division << 21;
                exp_out = exp_out - 21;
            end else if (division[1] == 1) begin
                division = division << 22;
                exp_out = exp_out - 22;
            end else if (division[0] == 1) begin
                division = division << 23;
                exp_out = exp_out - 23;
            end

            man_out = division[23:0];

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

    end

endmodule