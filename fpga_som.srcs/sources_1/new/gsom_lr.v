`timescale 1ns / 1ps

module gsom_lr
#(
    parameter DIGIT_DIM = 32,
    parameter ALPHA = 0.3,
    parameter R = 3.8,
    parameter LOG2_NODE_SIZE = 7    
)
(
    input wire [LOG2_NODE_SIZE:0] node_count,
    output wire [DIGIT_DIM-1:0]learning_rate
);
// 1- (r/ 2^(x/8 + 2))

    
endmodule
