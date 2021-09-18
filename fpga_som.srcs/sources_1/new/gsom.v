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
        parameter LOG2_NUM_CLASSES = 2,
        
        // model parameters
        parameter spread_factor = 32'h3F000000, //0.5
        parameter spread_factor_logval = 32'hBE9A209B, // BE9A209B = -0.30102999566
        
        parameter dimensions = 4,
        parameter dimensions_ieee754 = 32'h40800000, // 4
        parameter initial_learning_rate=32'h3E99999A, // 0.3
        parameter smooth_learning_factor= 32'h3F4CCCCD, //0.8
        parameter max_radius=4,
        parameter FD=0.1,
        parameter r=3.8,
        parameter alpha=0.9,
        parameter initial_node_size=30000

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
    reg [DIGIT_DIM-1:0] node_count_ieee754 = 32'h00000000;
    reg [LOG2_NODE_SIZE-1:0] map [ROWS-1:0][COLS-1:0];
    reg [(DIM*DIGIT_DIM)-1:0] node_list [MAX_NODE_SIZE-1:0];
    reg [LOG2_NODE_SIZE-1:0] node_coords [MAX_NODE_SIZE-1:0][1:0];
    reg [DIGIT_DIM-1:0] node_errors [MAX_NODE_SIZE-1:0];
    reg [DIGIT_DIM-1:0] growth_threshold;
    reg signed [3:0] radius;
    
    
    reg [DIGIT_DIM-1:0] learning_rate;
    reg [DIGIT_DIM-1:0] current_learning_rate;
    reg signed [LOG2_GROWING_ITERATIONS:0] iteration;
    
    reg signed [LOG2_TRAIN_ROWS:0] t1 = 0;
    reg signed [LOG2_TEST_ROWS:0] t2 = 0;
    
    reg init_gsom = 1;
    reg init_variables = 0;
    reg fit_en = 0;
    
    reg next_iteration_en = 0;
    reg next_t1_en = 0;
    
    reg mul_en = 0;
    reg mul_reset = 0;
    reg mul_num1;
    reg mul_num2;
    wire mul_num_out;
    wire mul_is_done;    
    
    fpa_multiplier multiplier(
        .clk(clk),
        .en(mul_en),
        .reset(mul_reset),
        .num1(mul_num1),
        .num2(mul_num2),
        .num_out(mul_num_out),
        .is_done(mul_is_done)
    );
    
    always @(posedge clk) begin
        if (init_gsom) begin
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
            
            node_count_ieee754 = 32'h40800000; // 4
            
            init_gsom = 0;
            init_variables = 1;
        end
        
        if (init_variables) begin
            learning_rate = initial_learning_rate;
            
            // growth threshold            
            mul_num1 = dimensions_ieee754;
            mul_num2 = dimensions_ieee754;
            mul_en = 1;
            mul_reset = 0;
            
            init_variables = 0;
        end
        
        if (mul_is_done) begin
            growth_threshold = mul_num_out;
            mul_en = 0;
            mul_reset = 1;
            fit_en = 1;
        end
    end
    
    reg get_LR_en = 0;
    
    always @(posedge clk) begin
        if (fit_en) begin
            current_learning_rate = learning_rate;
            iteration = -1;            
            next_iteration_en = 1;
        end
    end
    
    
    
    
    reg lr_en = 0;
    reg lr_reset = 0;
    reg [DIGIT_DIM-1:0] lr_node_count;
    reg [DIGIT_DIM-1:0] lr_prev_learning_rate;
    wire [DIGIT_DIM-1:0] lr_out;
    wire lr_is_done;
    
    gsom_learning_rate lr(
        .clk(clk), .en(lr_en), .reset(lr_reset),
        .node_count(lr_node_count),
        .prev_learning_rate(lr_prev_learning_rate),
        .alpha(alpha),
        .learning_rate(lr_out),
        .is_done(lr_is_done)
    );
        
    localparam delta_growing_iter = 25;
    localparam delta_smoothing_iter = 13;
    
    reg grow_en = 0;
    always @(posedge clk) begin
        if (next_iteration_en) begin
            if (iteration < GROWING_ITERATIONS) begin
                iteration = iteration + 1;
                // neighbourhood                
                if (iteration <= delta_growing_iter) begin
                    radius = 8;
                end else if ((iteration <= delta_growing_iter*2) && (iteration > delta_growing_iter*1)) begin
                    radius = 4;
                end else if ((iteration <= delta_growing_iter*3) && (iteration > delta_growing_iter*2)) begin
                    radius = 2;
                end else if ((iteration <= delta_growing_iter*4) && (iteration > delta_growing_iter*3)) begin
                    radius = 1;
                end
                
                // learning rate
                if (iteration != 0)
                    get_LR_en = 1;
                
                next_iteration_en = 0;
            end else begin
                iteration = -1;   // reset iteration count
                $finish;
            end
        end       
        
        // calculate learning rate
        if (get_LR_en) begin
            lr_en = 1;
            lr_reset = 0;
            lr_node_count = node_count_ieee754;
            lr_prev_learning_rate = current_learning_rate;
            
            get_LR_en = 0;
        end        
        if (lr_is_done) begin
            lr_en = 0;
            lr_reset = 1;
            current_learning_rate = lr_out;
            grow_en = 1;
        end
        
        // grow network
        if (grow_en) begin
            grow_en = 0;
            next_iteration_en = 1;
            // t1 = -1;
            // next_t1_en = 1;
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
