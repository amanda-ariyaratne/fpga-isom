`timescale 1ns / 1ps


module t(
input wire clk,
output wire is_done
    );

reg done=0;
assign is_done = done;

always @(posedge clk) begin
    done=0;
end

always @(negedge clk) begin
    done=1;
end
endmodule
