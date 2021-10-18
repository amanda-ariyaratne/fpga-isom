`timescale 1ns / 1ps

module gsom_grow_node_on_one_side
#(
    DIGIT_DIM=32,
    LOG2_NODE_COUNT=7
)
(
    input clk,
    input reset,
    input en,
    input [DIGIT_DIM-1:0] winner,
    input [DIGIT_DIM-1:0] node_next,
    output [DIGIT_DIM-1:0] weight,
    output is_done
);

reg done = 0;
reg init = 1;
reg [DIGIT_DIM-1:0] out;
reg add_en = 0;
reg add_reset = 0;
reg [DIGIT_DIM-1:0] winner_in;
reg [DIGIT_DIM-1:0] node_next_in;
wire [DIGIT_DIM-1:0] add_out;
wire add_is_done; 

fpa_adder adder(
    .clk(clk),
    .en(add_en),
    .reset(add_rpeset),
    .num1(winner_in),
    .num2(node_next_in),
    .num_out(add_out),
    .is_done(add_is_done)
);

reg comp_en=0;
reg comp_reset=0;
reg [DIGIT_DIM-1:0] comp_in;
wire [1:0] comp_out_max;
wire [1:0] comp_out_min;
wire comp_done_max;
wire comp_done_min;

fpa_comparator get_max(
    .clk(clk),
    .en(comp_en),
    .reset(comp_reset),
    .num1(32'h3F800000), //1
    .num2(comp_in),
    .num_out(comp_out_max),
    .is_done(comp_done_max)
);

fpa_comparator get_min(
    .clk(clk),
    .en(comp_en),
    .reset(comp_reset),
    .num1(comp_in), //0
    .num2(32'h00000000),
    .num_out(comp_out_min),
    .is_done(comp_done_min)
);

always @(posedge clk) begin
    if (en && init) begin
        winner_in = winner;
        winner_in[30:23] = winner[30:23]+1; // x2
        node_next_in = node_next;
        node_next_in[31] = ~node_next_in[31]; // -
        add_en=1;
        add_reset=0;
        init=0;
    end
    if (en && add_is_done) begin   
        add_en=0;
        add_reset=1;
             
        comp_en = 1;
        comp_reset = 0;
        comp_in = add_out;
    end
    if (en && comp_done_max && comp_done_min) begin
        comp_en=0;
        comp_reset=1;
        out = comp_in;
        done = 1;
        if (comp_out_max==2)
            out = 32'h3F800000;
        if (comp_out_min==2) // 0 is max
            out = 32'h00000000;
            
        if (comp_in!=out) $display("on_one_side Clipped 1 - %d 0 - %d", comp_in, out);
    end
end

always @(posedge reset) begin
    done = 0;
    init = 1;
end

assign is_done = done;
assign weight = out;

endmodule