`timescale 1ns / 1ps

module gsom_euclidean_distance
#(
    parameter DIM=4,
    parameter DIGIT_DIM=32,
    parameter LOG2_NODE_SIZE=14
)
(
    input wire clk,
    input wire en,
    input wire reset,
    input wire [DIGIT_DIM*DIM-1:0] weight,
    input wire [DIGIT_DIM*DIM-1:0] trainX,
    input wire [LOG2_NODE_SIZE-1:0] node_count,
    input wire [LOG2_NODE_SIZE-1:0] index,
    output wire [DIGIT_DIM-1:0] num_out,
    output wire is_done
);

////////////////////module enables and disbles///////////////////////
reg done=0;
reg init=1;                   
reg add_all_init=0;
reg wait_en = 0;
reg [DIGIT_DIM-1:0] out=0;
//////////////////////////////PART 1////////////////////////////
////////////////////distnace squer module///////////////////////
reg dist_sqr_en=0;
reg dist_sqr_reset=0;
wire [DIGIT_DIM*DIM-1:0] dist_sqr_in_1;
wire [DIGIT_DIM*DIM-1:0] dist_sqr_in_2;
wire [DIGIT_DIM*DIM-1:0] dist_sqr_out;
wire [DIGIT_DIM*DIM-1:0] dist_sqr_done;
wire [DIM-1:0] dist_sqr_is_done;


////////////////////Get all done to one variable///////////////////////
//genvar k;
//generate
//    for (k=1; k<=DIM; k=k+1) begin
//        assign dist_sqr_is_done[k-1] = dist_sqr_done[DIGIT_DIM*k-1];
//    end
//endgenerate

assign dist_sqr_is_done[1-1] = dist_sqr_done[DIGIT_DIM*1-1];
assign dist_sqr_is_done[2-1] = dist_sqr_done[DIGIT_DIM*2-1];
assign dist_sqr_is_done[3-1] = dist_sqr_done[DIGIT_DIM*3-1];
assign dist_sqr_is_done[4-1] = dist_sqr_done[DIGIT_DIM*4-1];

//assign dist_sqr_in_1 = weight;
//assign dist_sqr_in_2 = trainX;

////////////////////parallel blocks for distance square///////////////////////
genvar i;
generate
    for (i=DIGIT_DIM; i<=DIM*DIGIT_DIM; i=i+DIGIT_DIM) begin
        fpa_distance dist_sqr(
            .clk(clk),
            .en(dist_sqr_en),
            .reset(dist_sqr_reset),
            .num1(weight[i-1 -:DIGIT_DIM]),
            .num2(trainX[i-1 -:DIGIT_DIM]),
            .num_out(dist_sqr_out[i-1 -:DIGIT_DIM]),
            .is_done(dist_sqr_done[i-1])
        );
    end
endgenerate


////////////////////Calculate distance in parallel///////////////////////
always @(posedge clk) begin 
    if (en && init) begin
        init=0;
        if (node_count>index) begin
            dist_sqr_en=1;
            dist_sqr_reset=0;            
        end else
            done = 1;
     end
end

always @(posedge clk) begin 
    if (en && (dist_sqr_is_done == {(DIM){1'b1}})) begin
        dist_sqr_en=0;
        dist_sqr_reset=1;
        add_all_init=1;
    end
end

//////////////////////////////PART 2////////////////////////////
////////////////////Add all distance part///////////////////////
reg add_all_en=0;
reg add_all_reset=0;
reg [DIGIT_DIM-1:0] add_all_in_1=0;
reg [DIGIT_DIM-1:0] add_all_in_2;
wire [DIGIT_DIM-1:0] add_all_out;
wire add_all_done;

assign num_out = out;
assign is_done = done;

fpa_adder add_all(
    .clk(clk),
    .en(add_all_en),
    .reset(add_all_reset),
    .num1(add_all_in_1),
    .num2(add_all_in_2),
    .num_out(add_all_out),
    .is_done(add_all_done)
);

//////////////////////////calculate distance sum in parallel/////////////////////////
integer signed j=DIGIT_DIM;
always @(posedge clk) begin 
    if (add_all_init && !wait_en) begin
        add_all_in_2 = dist_sqr_out[j-1 -:DIGIT_DIM];
        add_all_en = 1;
        add_all_reset = 0;
        wait_en = 1;
    end
end

always @(posedge clk) begin 
    if (add_all_init && add_all_done) begin
        add_all_en = 0;
        add_all_in_1 = add_all_out;
        j = j + DIGIT_DIM;
        add_all_reset = 1;
        
        if (j > DIGIT_DIM*DIM) begin
            add_all_init = 0;
            done = 1;
            out = add_all_out;
        end else 
            wait_en = 0;
    end
end

//////////////////////////////reset block////////////////////////////
always @(posedge reset) begin
    done=0;
    init=1;
    wait_en = 0;
    add_all_init=0;
    
    dist_sqr_en=0;
    dist_sqr_reset=1;
    
    add_all_en=0;
    add_all_reset=1;
    add_all_in_1=0;
    
    j=DIGIT_DIM;    
end

endmodule