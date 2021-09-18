module gsom
    #(
        parameter DIM = 4,
        parameter LOG2_DIM = 3, 
        parameter DIGIT_DIM = 32,
        
        parameter INIT_ROWS = 2,
        parameter INIT_COLS = 2,
        
        parameter ROWS = 30,
        parameter LOG2_ROWS = 5, 
        parameter COLS = 30,
        parameter LOG2_COLS = 5,
        
        parameter MAX_NODE_SIZE = 100,
        parameter LOG2_NODE_SIZE = 7,
        
        parameter GROWING_ITERATIONS = 100,
        parameter LOG2_GROWING_ITERATIONS = 7,
        parameter SMOOTHING_ITERATIONS = 50,
        parameter LOG2_SMOOTHING_ITERATIONS = 6,
        
        parameter TRAIN_ROWS = 75,
        parameter LOG2_TRAIN_ROWS = 7,
        parameter TEST_ROWS = 150,
        parameter LOG2_TEST_ROWS = 8,
        
        parameter NUM_CLASSES = 3+1,
        parameter LOG2_NUM_CLASSES = 2

    )(
        input wire clk
    );
    
    reg [DIGIT_DIM*DIM-1:0] trainX [TRAIN_ROWS-1:0];    
    reg [DIGIT_DIM*DIM-1:0] testX [TEST_ROWS-1:0];
    reg [LOG2_NUM_CLASSES-1:0] trainY [TRAIN_ROWS-1:0];
    reg [LOG2_NUM_CLASSES-1:0] testY [TEST_ROWS-1:0];
    reg [LOG2_DIM*DIM-1:0] random_weights [INIT_ROWS-1:0];
    
    initial begin
        $readmemb("som_train_x.mem", trainX);
    end
    
    initial begin
        $readmemb("som_train_y.mem", trainY);
    end
    
    initial begin
        $readmemb("som_test_x.mem", testX);
    end
    
    initial begin
        $readmemb("som_test_y.mem", testY);
    end
    
    initial begin
        $readmemb("gsom_weights.mem", random_weights);
    end
    
    reg [LOG2_NODE_SIZE-1:0] node_count = 0;
    
    reg [(DIM*DIGIT_DIM)-1:0] node_list [MAX_NODE_SIZE-1:0];
    reg [LOG2_NODE_SIZE-1:0] map [ROWS-1:0][COLS-1:0];
    reg [DIGIT_DIM-1:0] node_errors [MAX_NODE_SIZE-1:0];
    reg [LOG2_NODE_SIZE-1:0] node_coords [MAX_NODE_SIZE-1:0][1:0];
    
    reg [DIGIT_DIM-1:0] learning_rate;
    reg [DIGIT_DIM-1:0] current_learning_rate;
    reg signed [LOG2_GROWING_ITERATIONS:0] iteration;
    
    reg signed [LOG2_TRAIN_ROWS:0] t1 = 0;
    reg signed [LOG2_TEST_ROWS:0] t2 = 0;
    
    reg init = 1;
    reg next_iteration_en = 0;
    reg next_t1_en = 0;
    
    always @(posedge clk) begin
        if (init) begin
            map[1][1] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 1;
            node_coords[node_count][1] = 1;
            node_count = node_count + 1;
            
            map[1][0] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 1;
            node_coords[node_count][1] = 0;
            node_count = node_count + 1;
            
            map[0][1] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 0;
            node_coords[node_count][1] = 1;
            node_count = node_count + 1;
            
            map[0][0] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 0;
            node_coords[node_count][1] = 0;
            node_count = node_count + 1;
            
            // initialize everything
            current_learning_rate = learning_rate;
            iteration = -1;
            
            next_iteration_en = 1;
            init = 0;
        end
    end
    
    reg calculate_lr_en = 0;
    
    always @(posedge clk) begin
        if (next_iteration_en) begin
            if (iteration < GROWING_ITERATIONS) begin
                iteration = iteration + 1;
                $display("iteration", iteration);         
                t1 = -1;
                next_t1_en = 1;
                next_iteration_en = 0;
            end else begin
                $finish;
            end
        end
    end
    
    always @(posedge clk) begin
        if (next_t1_en) begin
            if (t1 < TRAIN_ROWS) begin
                t1 = t1 + 1;   
                $display("t1", t1);         
            end else begin
                next_t1_en = 0;
                next_iteration_en = 1;
            end
        end
    end
    
endmodule
