`timescale 1ns / 1ps

module gsom
    #(
        parameter DIM = 4,
        parameter LOG2_DIM = 3, 
        parameter DIGIT_DIM = 32,
        
        parameter INIT_ROWS = 2,
        parameter INIT_COLS = 2,
        
        parameter MAX_ROWS = 10,
        parameter MAX_COLS = 10,
        
        parameter LOG2_ROWS = 7,         
        parameter LOG2_COLS = 7,

        parameter MAX_NODE_SIZE = 10,
        parameter LOG2_NODE_SIZE = 14,
        
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
        parameter spread_factor = 32'h3E99999A, //0.3
        parameter spread_factor_logval = 32'hBF9A1BC8, // BE9A209B = -1.20397280433
        
        parameter dimensions_ieee754 = 32'h40800000, // 4
        parameter initial_learning_rate=32'h3E99999A, // 0.3
        parameter smooth_learning_factor= 32'h3F4CCCCD, //0.8
        parameter max_radius=4,
        parameter FD=32'h3F8CCCCD, //1.1
        parameter r=32'h40733333, //3.8
        parameter alpha=32'h3F666666, //0.9
        parameter initial_node_size=10000

    )(
        input wire clk
    );
    
    ///////////////////////////////////////////**************************init variables**************************/////////////////////////////////////////////////////
    localparam [DIGIT_DIM-1:0] p_inf = 32'b0111_1111_1111_1111_1111_1111_1111_1111;
    localparam [DIGIT_DIM-1:0] n_inf = 32'b1111_1111_1111_1111_1111_1111_1111_1111;
    
    localparam delta_growing_iter = 34;
    localparam delta_smoothing_iter = 17;
    
    reg signed [LOG2_NODE_SIZE:0] i = 0;
    reg signed [LOG2_NODE_SIZE:0] j = 0;
    reg signed [LOG2_DIM:0] k = 0;
    
    reg [DIGIT_DIM*DIM-1:0] trainX [TRAIN_ROWS-1:0];    
    reg [DIGIT_DIM*DIM-1:0] testX [TEST_ROWS-1:0];
    reg [LOG2_NUM_CLASSES-1:0] trainY [TRAIN_ROWS-1:0];
    reg [LOG2_NUM_CLASSES-1:0] testY [TEST_ROWS-1:0];
    reg [DIGIT_DIM*DIM-1:0] random_weights [INIT_ROWS + INIT_COLS - 1:0];

    reg [LOG2_NODE_SIZE-1:0] node_count = 0;
    reg [DIGIT_DIM-1:0] node_count_ieee754 = 32'h00000000;
    reg [(DIM*DIGIT_DIM)-1:0] node_list [MAX_NODE_SIZE-1:0];
    reg signed [LOG2_NODE_SIZE-1:0] node_coords [MAX_NODE_SIZE-1:0][1:0];
    reg signed [((LOG2_NODE_SIZE+1)*4)-1:0] map [MAX_ROWS-1:0][MAX_COLS-1:0];
    reg [DIGIT_DIM-1:0] node_errors [MAX_NODE_SIZE-1:0];
    reg [DIGIT_DIM-1:0] growth_threshold;
    reg signed [3:0] radius;
    
    
    reg [DIGIT_DIM-1:0] learning_rate;
    reg [DIGIT_DIM-1:0] current_learning_rate;
    reg signed [LOG2_GROWING_ITERATIONS:0] iteration;
    
    reg signed [LOG2_TRAIN_ROWS:0] t1 = 0;
    reg signed [LOG2_TEST_ROWS:0] t2 = 0;
    
    reg init_arrays = 1;
    reg init_gsom = 0;
    reg init_variables = 0;
    reg square = 0;
    reg fit_en = 0;
    reg growing_iter_en = 0;
    reg smoothing_iter_en = 0;
    reg get_LR_en = 0;
    reg grow_en = 0;
    reg smooth_en = 0;
    reg adjust_weights_en=0;
    reg check_in_map_en = 0;
    reg smoothing_en=0;
    reg spread_weighs_en=0;
    reg grow_nodes_en=0;
    reg node_count_adder_en = 0;
    
    reg dist_enable = 0;
    reg init_neigh_search_en=0;  
    reg nb_search_en=0;
    reg not_man_dist_en=0;
    reg write_en = 0;
    reg is_completed = 0;
    reg min_distance_en=0;
    reg bmu_en=0;
    reg test_mode=0;
    reg classification_en=0;
    
    reg next_iteration_en = 0;
    reg next_t1_en = 0;
    
    reg [3:0] spreadable;
    reg signed [LOG2_NODE_SIZE:0] spreadable_idx [3:0];
    reg [3:0] grow_done=0;
    
    reg [DIGIT_DIM-1:0] min_distance;
    reg signed [LOG2_ROWS:0] minimum_distance_indices [MAX_NODE_SIZE-1:0][1:0];
    reg [LOG2_NODE_SIZE-1:0] minimum_distance_1D_indices [MAX_NODE_SIZE:0];
    reg [LOG2_NODE_SIZE-1:0] min_distance_next_index = 0;  
    integer node_count_i;
    
    reg signed [LOG2_ROWS:0] bmu [1:0];
    reg [LOG2_NODE_SIZE-1:0] rmu;
    
    reg signed [LOG2_ROWS:0] bmu_i;
    reg signed [LOG2_COLS:0] bmu_j;
    
    reg signed [LOG2_ROWS:0] abs_bmu_i;
    reg signed [LOG2_COLS:0] abs_bmu_j;
    
    reg [1:0] i_j_signs;
    
    reg signed [LOG2_NODE_SIZE:0] rmu_i;
    
    reg signed [LOG2_ROWS:0] bmu_x;
    reg signed [LOG2_COLS:0] bmu_y;
    reg signed [DIGIT_DIM-1 :0] man_dist;
    
    reg signed [LOG2_ROWS:0] leftx, rightx, upx, bottomx;
    reg signed [LOG2_COLS:0] lefty, righty, upy, bottomy;
    
    reg [LOG2_NODE_SIZE:0] node_iter = 0;
    
    reg signed [LOG2_NODE_SIZE:0] u_idx;
    reg signed [LOG2_NODE_SIZE:0] r_idx;
    reg signed [LOG2_NODE_SIZE:0] b_idx;
    reg signed [LOG2_NODE_SIZE:0] l_idx;  
    
    reg check_valid_idx = 0;  
    
    reg signed [LOG2_ROWS:0] new_node_idx_x [3:0];
    reg signed [LOG2_COLS:0] new_node_idx_y [3:0];
    ///////////////////////////////////////////**************************read input**************************/////////////////////////////////////////////////////
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
    
    ///////////////////////////////////////////**************************components**************************/////////////////////////////////////////////////////
    ///////////////////////////////////////////**************************multiplier**************************/////////////////////////////////////////////////////

    reg mul_en = 0;
    reg mul_reset = 0;
    reg [DIGIT_DIM-1:0] mul_num1;
    reg [DIGIT_DIM-1:0] mul_num2;
    wire [DIGIT_DIM-1:0] mul_num_out;
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
    
    ///////////////////////////////////////////**************************node_count_adder**************************/////////////////////////////////////////////////////

    reg nca_en = 0;
    reg nca_reset = 0;
    reg [DIGIT_DIM-1:0] nca_num1;
    reg [DIGIT_DIM-1:0] nca_num2;
    wire [DIGIT_DIM-1:0] nca_num_out;
    wire nca_is_done;    
    
    fpa_adder nca(
        .clk(clk),
        .en(nca_en),
        .reset(nca_reset),
        .num1(nca_num1),
        .num2(nca_num2),
        .num_out(nca_num_out),
        .is_done(nca_is_done)
    );
    ///////////////////////////////////////////**************************learning_rate**************************/////////////////////////////////////////////////////
    reg lr_en = 0;
    reg lr_reset = 0;
    wire [DIGIT_DIM-1:0] lr_out;
    wire lr_is_done;
    
    gsom_learning_rate lr(
        .clk(clk), .en(lr_en), .reset(lr_reset),
        .node_count(node_count_ieee754),
        .prev_learning_rate(current_learning_rate),
        .alpha(alpha),
        .learning_rate(lr_out),
        .is_done(lr_is_done)
    );
    ///////////////////////////////////////////**************************euclidean_distance**************************/////////////////////////////////////////////////////
    wire [DIGIT_DIM-1:0] distance_out [MAX_NODE_SIZE-1:0];
    wire [MAX_NODE_SIZE-1:0] distance_done;
    reg distance_en;
    reg distance_reset;
    reg [DIGIT_DIM*DIM-1:0] distance_X;

    genvar euc_i;
    generate
        for(euc_i=0; euc_i<=MAX_NODE_SIZE-1; euc_i=euc_i+1) begin
            gsom_euclidean_distance euclidean_distance(
                .clk(clk),
                .en(distance_en),
                .reset(distance_reset),
                .weight(node_list[euc_i]),
                .trainX(distance_X),
                .node_count(node_count),
                .index(euc_i[LOG2_NODE_SIZE-1:0]),
                .num_out(distance_out[euc_i]),
                .is_done(distance_done[euc_i])
            );
        end
    endgenerate
    ///////////////////////////////////////////**************************get_min**************************/////////////////////////////////////////////////////
    reg [DIGIT_DIM-1:0] comp_in_1;
    reg [DIGIT_DIM-1:0] comp_in_2;
    wire [1:0] comp_out;
    wire comp_done;
    reg comp_en=0;
    reg comp_reset=0;
    
    fpa_comparator get_min(
        .clk(clk),
        .en(comp_en),
        .reset(comp_reset),
        .num1(comp_in_1),
        .num2(comp_in_2),
        .num_out(comp_out),
        .is_done(comp_done)
    );
    ///////////////////////////////////////////**************************update_bmu & update_neighbour**************************/////////////////////////////////////////////////////
    reg update_en=0;
    reg update_reset=0;
    reg [DIGIT_DIM*DIM-1:0] update_in_1;
    reg [DIGIT_DIM*DIM-1:0] update_in_2;
    reg [DIGIT_DIM-1:0]  update_learning_rate;
    wire [DIGIT_DIM*DIM-1:0] update_out;
    wire [DIM-1:0] update_done;

    reg update_neighbour_en=0;
    reg update_neighbour_reset=0;
    reg [DIGIT_DIM*DIM-1:0] update_neighbour_in_1;
    reg [DIGIT_DIM*DIM-1:0] update_neighbour_in_2;
    reg [DIGIT_DIM-1:0]  update_neighbour_learning_rate;
    reg [DIGIT_DIM-1:0]  update_neighbour_man_dist;
    wire [DIGIT_DIM*DIM-1:0] update_neighbour_out;
    wire [DIM-1:0] update_neighbour_done;
    
    genvar update_i;
    generate
        for (update_i=1; update_i<=DIM; update_i=update_i+1) begin
            gsom_update_weight update_weight(
                .clk(clk),
                .en(update_en),
                .reset(update_reset),
                .weight(update_in_1[update_i*DIGIT_DIM-1 -:DIGIT_DIM]),
                .train_row(update_in_2[update_i*DIGIT_DIM-1 -:DIGIT_DIM]),
                .alpha(update_learning_rate),
                .updated_weight(update_out[update_i*DIGIT_DIM-1 -:DIGIT_DIM]),
                .is_done(update_done[update_i-1])
            );
            
            gsom_update_neighbour update_neighbour_weight(
                .clk(clk),
                .en(update_neighbour_en),
                .reset(update_neighbour_reset),
                .weight(update_neighbour_in_1[update_i*DIGIT_DIM-1 -:DIGIT_DIM]),
                .neighbour(update_neighbour_in_2[update_i*DIGIT_DIM-1 -:DIGIT_DIM]),
                .learning_rate(update_neighbour_learning_rate),
                .man_dist(update_neighbour_man_dist),
                .num_out(update_neighbour_out[update_i*DIGIT_DIM-1 -:DIGIT_DIM]),
                .is_done(update_neighbour_done[update_i-1])
            );
        end
    endgenerate
    ///////////////////////////////////////////**************************update_node_error**************************/////////////////////////////////////////////////////
    reg node_error_add_en=0;
    reg node_error_add_reset;
    reg [31:0] node_error_add_in_1;
    reg [31:0] node_error_add_in_2;
    wire [31:0] node_error_add_out;
    wire node_error_add_done;
    
    fpa_adder node_error_adder(
        .clk(clk),
        .en(node_error_add_en),
        .reset(node_error_add_reset),
        .num1(node_error_add_in_1),
        .num2(node_error_add_in_2),
        .num_out(node_error_add_out),
        .is_done(node_error_add_done)
    );
    
    ///////////////////////////////////////////**************************update_node_error_spreading**************************/////////////////////////////////////////////////////
    reg update_error_en = 0;
    reg update_error_reset = 0;
    wire [DIGIT_DIM-1:0] updated_error [3:0];
    wire [3:0] update_error_done;    
    
    genvar error_i;
    generate 
        for(error_i=0;error_i<4;error_i=error_i+1) begin // 4 = up + down + left + right
            fpa_multiplier update_error(
                .clk(clk),
                .en(update_error_en),
                .reset(update_error_reset),
                .num1(node_errors[spreadable_idx[error_i]]),
                .num2(FD),
                .num_out(updated_error[error_i]),
                .is_done(update_error_done[error_i])
            );
        end
    endgenerate
    
    ///////////////////////////////////////////**************************new_node_in_middle**************************/////////////////////////////////////////////////////
    reg [DIM*4-1:0] new_node_in_middle_en = 0;
    reg [DIM*4-1:0] new_node_in_middle_reset = 0;
    reg [DIGIT_DIM*DIM*4-1:0] new_node_in_middle_winner, new_node_in_middle_next_node;
    wire [DIM*4-1:0] new_node_in_middle_is_done;
    wire [DIGIT_DIM*DIM*4-1:0] new_node_in_middle_weight;

    genvar nim_i, nim_dim_i;
    for (nim_i=0; nim_i<4; nim_i=nim_i+1) begin
        for (nim_dim_i=0; nim_dim_i<DIM; nim_dim_i=nim_dim_i+1) begin
            gsom_grow_node_in_middle grow_node_in_middle(
                .clk(clk),
                .reset(new_node_in_middle_reset[nim_i*4+nim_dim_i]),
                .en(new_node_in_middle_en[nim_i*4+nim_dim_i]),
                .winner(new_node_in_middle_winner[DIGIT_DIM*DIM*(nim_i)+DIGIT_DIM*(nim_dim_i+1) -1 -:DIGIT_DIM]),
                .node_next(new_node_in_middle_next_node[DIGIT_DIM*DIM*(nim_i)+DIGIT_DIM*(nim_dim_i+1) -1 -:DIGIT_DIM]),
                .weight(new_node_in_middle_weight[DIGIT_DIM*DIM*(nim_i)+DIGIT_DIM*(nim_dim_i+1) -1 -:DIGIT_DIM]),
                .is_done(new_node_in_middle_is_done[nim_i*4+nim_dim_i])
            );
        end
    end
    
    ///////////////////////////////////////////**************************new_node_on_one_side**************************/////////////////////////////////////////////////////
    reg [DIM*4-1:0] new_node_on_one_side_en = 0;
    reg [DIM*4-1:0] new_node_on_one_side_reset = 0;
    reg [DIGIT_DIM*DIM*4-1:0] new_node_on_one_side_winner, new_node_on_one_side_next_node;
    wire [DIM*4-1:0] new_node_on_one_side_is_done;
    wire [DIGIT_DIM*DIM*4-1:0] new_node_on_one_side_weight;
    
    genvar noos_i, noos_dim_i;
    for (noos_i=0; noos_i<4; noos_i=noos_i+1) begin
        for (noos_dim_i=0; noos_dim_i<DIM; noos_dim_i=noos_dim_i+1) begin
            gsom_grow_node_on_one_side grow_node_on_one_side(
                .clk(clk),
                .reset(new_node_on_one_side_reset[noos_i*4+noos_dim_i]),
                .en(new_node_on_one_side_en[noos_i*4+noos_dim_i]),
                .winner(new_node_on_one_side_winner[DIGIT_DIM*DIM*(noos_i)+DIGIT_DIM*(noos_dim_i+1) -1 -:DIGIT_DIM]),
                .node_next(new_node_on_one_side_next_node[DIGIT_DIM*DIM*(noos_i)+DIGIT_DIM*(noos_dim_i+1) -1 -:DIGIT_DIM]),
                .weight(new_node_on_one_side_weight[DIGIT_DIM*DIM*(noos_i)+DIGIT_DIM*(noos_dim_i+1) -1 -:DIGIT_DIM]),
                .is_done(new_node_on_one_side_is_done[noos_i*4+noos_dim_i])
            );
        end
    end
    
    ///////////////////////////////////////////**************************end_module_initialization**************************/////////////////////////////////////////////////////

    ///////////////////////////////////////////**************************check_is_in_map**************************/////////////////////////////////////////////////////
    function signed [LOG2_NODE_SIZE:0] is_in_map;
        input signed [LOG2_ROWS:0] x;
        input signed [LOG2_COLS:0] y;
        reg [LOG2_ROWS-1:0] abs_x;
        reg [LOG2_COLS-1:0] abs_y;
        reg [2:0] x_y_signs;
        
        begin
            x_y_signs=0;
            abs_x = x>0 ? x : -x;
            abs_y = y>0 ? y : -y;
            
            x_y_signs[0] = x[LOG2_ROWS];
            x_y_signs[1] = y[LOG2_COLS];
//            if (t1==9 && bmu_i==bmu[1] && bmu_j==bmu[0]) $display("x_y_signs - %d, 0 - %d, 1 - %d, -1 - %d, 2 - %d", 
//                x_y_signs,
//                map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*1-1 -:(LOG2_NODE_SIZE+1)],
//                map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*2-1 -:(LOG2_NODE_SIZE+1)],
//                map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*3-1 -:(LOG2_NODE_SIZE+1)],
//                map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*4-1 -:(LOG2_NODE_SIZE+1)]
//            );
            
            if ((map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*1-1 -:(LOG2_NODE_SIZE+1)] != {LOG2_NODE_SIZE+1{1'b1}}) && (x_y_signs==0)) begin
                is_in_map = map[abs_x][abs_y][LOG2_NODE_SIZE*1-1 -:LOG2_NODE_SIZE];
                
            end else if ((map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*2-1 -:(LOG2_NODE_SIZE+1)] != {LOG2_NODE_SIZE+1{1'b1}}) && (x_y_signs==1)) begin
                is_in_map = map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*2-1 -:(LOG2_NODE_SIZE+1)];
                
            end else if ((map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*3-1 -:(LOG2_NODE_SIZE+1)] != {LOG2_NODE_SIZE+1{1'b1}}) && (x_y_signs==2)) begin
                is_in_map = map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*3-1 -:(LOG2_NODE_SIZE+1)];
                
            end else if ((map[abs_x][abs_y][LOG2_NODE_SIZE*4-1 -:LOG2_NODE_SIZE] != {LOG2_NODE_SIZE+1{1'b1}}) && (x_y_signs==3)) begin
                is_in_map = map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*4-1 -:(LOG2_NODE_SIZE+1)];
                
            end else begin
                is_in_map = -1;
            end
            
//            if (t1==39 && bmu_i==bmu[1] && bmu_j==bmu[0]) $display("is_in_map %d", is_in_map);
        end
    endfunction
    
    ///////////////////////////////////////////**************************insert_new_node**************************/////////////////////////////////////////////////////
    task insert_new_node;
        input signed [LOG2_ROWS:0] x;
        input signed [LOG2_COLS:0] y;
        input [DIGIT_DIM*DIM-1:0] weights;
        reg [LOG2_ROWS-1:0] abs_x;
        reg [LOG2_COLS-1:0] abs_y;
        reg [2:0] x_y_signs;
        
        begin
            x_y_signs=0;
            abs_x = x>0 ? x : -x;
            abs_y = y>0 ? y : -y;
            
            x_y_signs[0] = x[LOG2_ROWS];
            x_y_signs[1] = y[LOG2_COLS];
            
            map[abs_x][abs_y][(LOG2_NODE_SIZE+1)*(x_y_signs+1)-1 -:(LOG2_NODE_SIZE+1)] = node_count;
            node_list[node_count] = weights;
            node_coords[node_count][1] = x;
            node_coords[node_count][0] = y;
            node_count = node_count + 1;  
            node_count_adder_en=1;
        end
    endtask

    ///////////////////////////////////////////**************************init_arrays**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (init_arrays) begin
            for (i = 0; i < MAX_ROWS; i = i + 1) begin
                for (j = 0; j < MAX_COLS; j = j + 1) begin
                    map[i][j] = { (LOG2_NODE_SIZE+1)*4{1'b1} }; // initialize all cells to -1
                end    
            end
            
            for (node_iter = 0; node_iter < MAX_NODE_SIZE; node_iter = node_iter + 1) begin
                node_errors[node_iter] = 0;  
            end
            init_arrays = 0;
            init_gsom = 1;
        end
    end
    
    ///////////////////////////////////////////**************************init_gsom**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (init_gsom) begin
            map[1][1][LOG2_NODE_SIZE:0] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 1;
            node_coords[node_count][1] = 1;
            node_count = node_count + 1;            
            
            map[1][0][LOG2_NODE_SIZE:0] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 1;
            node_coords[node_count][1] = 0;
            node_count = node_count + 1;
            
            map[0][1][LOG2_NODE_SIZE:0] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 0;
            node_coords[node_count][1] = 1;
            node_count = node_count + 1;
            
            map[0][0][LOG2_NODE_SIZE:0] = node_count;
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
            mul_num1 = {1'b1, dimensions_ieee754[DIGIT_DIM-2:0]};
            mul_num2 = spread_factor_logval;
            mul_en = 1;
            mul_reset = 0;
            
            if (mul_is_done) begin
                growth_threshold = mul_num_out;
                mul_en = 0;
                mul_reset = 1;
                
                init_variables = 0;
                square = 1;
            end
        end
        if (square && !mul_is_done) begin
            mul_num1 = mul_num_out;
            //mul_num2 = 32'h3F800000; //1
            mul_num2 = mul_num_out; //1
            mul_en = 1;
            mul_reset = 0;
        end
        if (square && mul_is_done && mul_en) begin
            growth_threshold = mul_num_out;
            mul_en = 0;
            mul_reset = 1;
            fit_en = 1;
            growing_iter_en = 1;
            square = 0;
        end
    end
    
    ///////////////////////////////////////////**************************growing_smoothing_iter_en_init**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (fit_en && growing_iter_en) begin
            current_learning_rate = learning_rate;
            iteration = -1;            
            next_iteration_en = 1;
            fit_en = 0;
        end
        
        if (fit_en && smoothing_iter_en) begin
            mul_num1 = learning_rate;
            mul_num2 = smooth_learning_factor;
            mul_en = 1;
            mul_reset = 0;
            if (mul_is_done) begin
                current_learning_rate = mul_num_out;
                mul_en = 0;
                mul_reset = 1;
                iteration = -1;            
                next_iteration_en = 1;
                fit_en = 0;
            end
        end
    end
    
    ///////////////////////////////////////////**************************growing_iterations**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (next_iteration_en && growing_iter_en) begin
            if (iteration < GROWING_ITERATIONS) begin
                iteration = iteration + 1;
//                $display("iteration", iteration);
                
                // neighbourhood                
                if (iteration <= delta_growing_iter) begin
                    radius = 4;
                end else if ((iteration <= delta_growing_iter*2) && (iteration > delta_growing_iter*1)) begin
                    radius = 2;
                end else if ((iteration <= delta_growing_iter*3) && (iteration > delta_growing_iter*2)) begin
                    radius = 1;
                end 
                
                // learning rate
                if (iteration != 0) begin
                    get_LR_en = 1;
                end
                if (iteration == 0) begin
                    current_learning_rate = learning_rate;
                    grow_en = 1;
                end
            end else begin
                fit_en = 1;
                smoothing_iter_en = 1;                
                growing_iter_en = 0;
            end
            next_iteration_en = 0;
        end
 
        // calculate learning rate
        if (get_LR_en && growing_iter_en) begin
            lr_reset = 0;
            lr_en = 1;
            if (lr_is_done) begin
                lr_en = 0;
                lr_reset = 1;
                current_learning_rate = lr_out;
                get_LR_en = 0;
                grow_en = 1;
            end
        end
        
        // grow network
        if (grow_en) begin
            grow_en = 0;
            t1 = -1;
            next_t1_en = 1;
        end
    end
    
    ///////////////////////////////////////////**************************smoothing_iterations**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (next_iteration_en && smoothing_iter_en) begin
            if (iteration < SMOOTHING_ITERATIONS) begin
                iteration = iteration + 1;
//                $display("iteration", iteration);
                // neighbourhood                
                if (iteration <= delta_smoothing_iter) begin
                    radius = 4;                
                end else if ((iteration <= delta_smoothing_iter*2) && (iteration > delta_smoothing_iter*1)) begin
                    radius = 2;
                end else if ((iteration <= delta_smoothing_iter*3) && (iteration > delta_smoothing_iter*2)) begin
                    radius = 1;
                end 
                
                // learning rate
                if (iteration != 0)
                    get_LR_en = 1;
                    
                if (iteration == 0) begin
                    smooth_en = 1;
                end
                
                next_iteration_en = 0;
            end else begin
                iteration = -1;
                // write_en = 1;
                is_completed = 1;
                smoothing_iter_en = 0;
            end
        end       
        
        // calculate learning rate
        if (get_LR_en && smoothing_iter_en) begin
            lr_en = 1;
            lr_reset = 0;
            if (lr_is_done) begin
                lr_en = 0;
                lr_reset = 1;
                current_learning_rate = lr_out;
                get_LR_en = 0;
                smooth_en = 1;
            end
        end        
        
        if (smooth_en) begin
            smooth_en = 0;
            t1 = -1;
            next_t1_en = 1;
        end
    end   
    
    always @(posedge clk) begin
        if (next_t1_en) begin
            if (t1 < TRAIN_ROWS-1) begin
                t1 = t1 + 1;   
//                $display("iter %d t1 %d node count %d", iteration, t1, node_count);
                
                dist_enable = 1;
            end else begin
                next_iteration_en = 1;
            end
            next_t1_en = 0;
        end
    end
    
    ///////////////////////////////////////////**************************Find_BMU**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (dist_enable) begin
            distance_X=trainX[t1];
            distance_reset=0;
            distance_en=1;
            if (distance_done == {MAX_NODE_SIZE{1'b1}}) begin
                distance_en = 0;
                distance_reset = 1;
                node_count_i = 0;
                min_distance_next_index = 0;
                min_distance = p_inf;
                dist_enable = 0;
                min_distance_en = 1;
            end
        end
    end
    
    always @(posedge clk) begin
        if (min_distance_en) begin
            comp_in_1 = min_distance;
            comp_in_2 = distance_out[node_count_i];
            comp_reset = 0;
            comp_en = 1;
            
            if (comp_done) begin
                comp_en = 0;
                comp_reset = 1;
                
                if (comp_out==0) begin
                    minimum_distance_indices[min_distance_next_index][1] = node_coords[node_count_i][1];
                    minimum_distance_indices[min_distance_next_index][0] = node_coords[node_count_i][0];
                    minimum_distance_1D_indices[min_distance_next_index] = node_count_i;
                    min_distance_next_index = min_distance_next_index + 1;
                
                end else if (comp_out==1) begin
                    min_distance = distance_out[node_count_i];
                    minimum_distance_indices[0][1] = node_coords[node_count_i][1];
                    minimum_distance_indices[0][0] = node_coords[node_count_i][0]; 
                    minimum_distance_1D_indices[0] = node_count_i;                   
                    min_distance_next_index = 1;
                end 
                
                node_count_i=node_count_i+1;
                if (node_count_i>=node_count) begin
                    min_distance_en=0;
                    bmu_en=1;
                end
            end
        end
    end
    
    always @(posedge clk) begin
        if (bmu_en && !test_mode) begin   
            bmu[1] = minimum_distance_indices[min_distance_next_index-1][1];
            bmu[0] = minimum_distance_indices[min_distance_next_index-1][0];
            rmu = minimum_distance_1D_indices[0];
            
            if (!classification_en)
                init_neigh_search_en = 1;
            else
                next_t1_en = 1;
            bmu_en=0;
        end
    end
    
    ///////////////////////////////////////////**************************init_neigh_search_en**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin    
        if (init_neigh_search_en) begin
            bmu_x = bmu[1]; bmu_y = bmu[0];  
            bmu_i = bmu_x-radius;            
            bmu_j = bmu_y-radius;
            init_neigh_search_en=0;
            nb_search_en=1;
//            $display("bmu %d %d", bmu[1], bmu[0]);
        end
    end
    
    always @(posedge clk) begin    
        if (nb_search_en && !update_en && !update_neighbour_en) begin  
            man_dist = (bmu_x-bmu_i) >= 0 ? (bmu_x-bmu_i) : (bmu_i-bmu_x);
            man_dist = man_dist + ((bmu_y - bmu_j)>= 0 ? (bmu_y - bmu_j) : (bmu_j - bmu_y));   
            
            if (man_dist == 0) begin
                //$display("bmu_weight_update %d %d", bmu_i, bmu_j);
                update_in_1 = node_list[rmu];
                update_in_2 = trainX[t1];
                update_learning_rate = current_learning_rate;
                update_en=1;
                update_reset=0;
                
                // node error adder  
                node_error_add_in_1 = node_errors[rmu];
                node_error_add_in_2 = min_distance;
                node_error_add_reset = 0;
                node_error_add_en = 1;  

            end else if (man_dist <= radius) begin
                //$display("neighbour_update %d %d", bmu_i, bmu_j);
                update_neighbour_in_1 = node_list[rmu];
                update_neighbour_in_2 = node_list[rmu_i];
                update_neighbour_learning_rate = current_learning_rate;
                update_neighbour_man_dist = man_dist;
                update_neighbour_en=1; 
                update_neighbour_reset=0;             
            end else begin
                not_man_dist_en = 1;
            end
            nb_search_en = 0;
        end
    end
    
    always @(posedge clk) begin
        if ((update_done == {DIM{1'b1}}) || (update_neighbour_done == {DIM{1'b1}}) || not_man_dist_en) begin
            if (update_done == {DIM{1'b1}} && node_error_add_done) begin
                node_error_add_en = 0;
                node_error_add_reset = 1;
                node_errors[rmu] = node_error_add_out;
                
                update_en=0;
                update_reset=1;
                node_list[rmu] = update_out;
                if (growing_iter_en)
                    adjust_weights_en = 1;
            end 
            if (update_neighbour_done == {DIM{1'b1}}) begin
                node_list[rmu_i] = update_neighbour_out;
                update_neighbour_en=0;
                update_neighbour_reset=1;
            end      
            if (!adjust_weights_en && !update_en && !update_neighbour_en) begin
                bmu_j = bmu_j + 1;
                if (bmu_j == bmu_y+radius+1) begin
                    bmu_j = bmu_y-radius;                
                    bmu_i = bmu_i + 1;
                    if (bmu_i == bmu_x+radius+1) begin
                        nb_search_en = 0; // neighbourhood search finished        
                        next_t1_en = 1; // go to the next input
                    end else
                        nb_search_en = 1; // next neighbour
                end else begin
                    nb_search_en = 1; // next neighbour
                end
                not_man_dist_en = 0; // close this block
                check_valid_idx = 1;
            
                if (nb_search_en && check_valid_idx) begin
                    rmu_i = is_in_map(bmu_i, bmu_j);
                    if (rmu_i != -1) begin
                        nb_search_en = 1;
                    end else begin
//                        if (bmu[1]==bmu_i && bmu[0]==bmu_j)
//                            $finish;
                        nb_search_en = 0;
                        not_man_dist_en = 1;
                    end
                    check_valid_idx = 0;
                end
            end
        end
    end
    
    ///////////////////////////////////////////**************************adjust_weights**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (adjust_weights_en) begin
            comp_reset = 0;
            comp_en = 1;
            comp_in_1 = node_errors[rmu];
            comp_in_2 = growth_threshold;
        
            if (comp_done) begin
                comp_reset = 1;
                comp_en = 0;
                if (comp_out == 1) begin
                    leftx = bmu_x-1;    lefty = bmu_y;
                    rightx = bmu_x+1;   righty = bmu_y;
                    upx = bmu_x;        upy = bmu_y+1;
                    bottomx = bmu_x;    bottomy = bmu_y-1;
                    
                    spreadable_idx[0] = is_in_map(upx, upy);
                    spreadable[0] = spreadable_idx[0]==-1 ? 0 : 1;
                    
                    spreadable_idx[1] = is_in_map(rightx, righty);
                    spreadable[1] = spreadable_idx[1]==-1 ? 0 : 1;
                    
                    spreadable_idx[2] = is_in_map(bottomx, bottomy);
                    spreadable[2] = spreadable_idx[2]==-1 ? 0 : 1;
                    
                    spreadable_idx[3] = is_in_map(leftx, lefty);
                    spreadable[3] = spreadable_idx[3]==-1 ? 0 : 1;
                        
                    if (spreadable == {4{1'b1}}) begin
//                        $display("spreadable");
                        spread_weighs_en=1;
                    end else begin      
//                        $display("grow_nodes_en");   
                        grow_nodes_en=1;
                        grow_done = spreadable;
                    end
                    
                end else begin
                    not_man_dist_en = 1;
//                    $display("GT is bigger");   
                end
                adjust_weights_en = 0;
            end 
        end
        
        if (spread_weighs_en) begin
            node_errors[rmu] = growth_threshold;
            node_errors[rmu][30:23] = growth_threshold[30:23] - 1; // divide by 2 => exp-1
            update_error_en = 1;
            update_error_reset = 0;
            if (update_error_done == {4{1'b1}}) begin
                update_error_en = 0;
                update_error_reset = 1;
                node_errors[spreadable_idx[0]] = updated_error[0];
                node_errors[spreadable_idx[1]] = updated_error[1];
                node_errors[spreadable_idx[2]] = updated_error[2];
                node_errors[spreadable_idx[3]] = updated_error[3]; 
                
                not_man_dist_en = 1;   
                spread_weighs_en = 0;   
            end
        end
    end

    ///////////////////////////////////////////**************************grow_up**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (grow_nodes_en && !spreadable[0]) begin
//            $display("new up node %d %d", upx, upy);
            new_node_idx_x[0] = upx;
            new_node_idx_y[0] = upy;
            u_idx = is_in_map(upx, upy+1);
            if (u_idx != -1) begin
//                $display("UP 1");
                new_node_in_middle_en[1*DIM -:DIM] = {DIM{1'b1}};
                new_node_in_middle_reset[1*DIM -:DIM] = 0;
                new_node_in_middle_winner[DIGIT_DIM*DIM*1-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                new_node_in_middle_next_node[DIGIT_DIM*DIM*1-1 -:DIGIT_DIM*DIM] = node_list[u_idx];
                
            end else begin
                u_idx = is_in_map(bmu[1], bmu[0]-1);
                if (u_idx != -1) begin
//                    $display("UP 2");
                    new_node_on_one_side_en[1*DIM-1 -:DIM] = {DIM{1'b1}};
                    new_node_on_one_side_reset[1*DIM-1 -:DIM] = 0;
                    new_node_on_one_side_winner[DIGIT_DIM*DIM*1-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                    new_node_on_one_side_next_node[DIGIT_DIM*DIM*1-1 -:DIGIT_DIM*DIM] = node_list[u_idx];
                    
//                    $display(" winner %h %h %h %h next %h %h %h %h", 
//                        new_node_on_one_side_winner[DIGIT_DIM*4*1-1 -:DIGIT_DIM],
//                        new_node_on_one_side_winner[DIGIT_DIM*3*1-1 -:DIGIT_DIM],
//                        new_node_on_one_side_winner[DIGIT_DIM*2*1-1 -:DIGIT_DIM],
//                        new_node_on_one_side_winner[DIGIT_DIM*1*1-1 -:DIGIT_DIM],
//                        new_node_on_one_side_next_node[DIGIT_DIM*4*1-1 -:DIGIT_DIM],
//                        new_node_on_one_side_next_node[DIGIT_DIM*3*1-1 -:DIGIT_DIM],
//                        new_node_on_one_side_next_node[DIGIT_DIM*2*1-1 -:DIGIT_DIM],
//                        new_node_on_one_side_next_node[DIGIT_DIM*1*1-1 -:DIGIT_DIM],
                        
//                    );
                    
                end else begin
                    u_idx = is_in_map(bmu[1]+1, bmu[0]);
                    if (u_idx != -1) begin
//                        $display("UP 3");
                        new_node_on_one_side_en[1*DIM-1 -:DIM] = {DIM{1'b1}};
                        new_node_on_one_side_reset[1*DIM-1 -:DIM] = 0;
                        new_node_on_one_side_winner[DIGIT_DIM*DIM*1-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                        new_node_on_one_side_next_node[DIGIT_DIM*DIM*1-1 -:DIGIT_DIM*DIM] = node_list[u_idx];
                        
                    end else begin
                        u_idx = is_in_map(bmu[1]-1, bmu[0]);
                        if (u_idx != -1) begin
//                            $display("UP 4");
                            new_node_on_one_side_en[1*DIM-1 -:DIM] = {DIM{1'b1}};
                            new_node_on_one_side_reset[1*DIM-1 -:DIM] = 0;
                            new_node_on_one_side_winner[DIGIT_DIM*DIM*1-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                            new_node_on_one_side_next_node[DIGIT_DIM*DIM*1-1 -:DIGIT_DIM*DIM] = node_list[u_idx];
                        end 
                    end
                end
            end            
            spreadable[0] = 1;
        end 
    end
    
    ///////////////////////////////////////////**************************grow_right**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (grow_nodes_en && !spreadable[1]) begin
//            $display("new right node %d %d", rightx, righty);
            new_node_idx_x[1] = rightx;
            new_node_idx_y[1] = righty;
            r_idx = is_in_map(rightx+1, righty);
            if (r_idx != -1) begin
                new_node_in_middle_en[2*DIM-1 -:DIM] = {DIM{1'b1}};
                new_node_in_middle_reset[2*DIM-1 -:DIM] = 0;
                new_node_in_middle_winner[DIGIT_DIM*DIM*2-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                new_node_in_middle_next_node[DIGIT_DIM*DIM*2-1 -:DIGIT_DIM*DIM] = node_list[r_idx];
            end else begin
                r_idx = is_in_map(bmu[1]-1, bmu[0]);
                if (r_idx != -1) begin
                    new_node_on_one_side_en[2*DIM-1 -:DIM] = {DIM{1'b1}};
                    new_node_on_one_side_reset[2*DIM-1 -:DIM] = 0;
                    new_node_on_one_side_winner[DIGIT_DIM*DIM*2-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                    new_node_on_one_side_next_node[DIGIT_DIM*DIM*2-1 -:DIGIT_DIM*DIM] = node_list[r_idx];
                    
                end else begin
                    r_idx = is_in_map(bmu[1], bmu[0]+1);
                    if (r_idx != -1) begin
                        new_node_on_one_side_en[2*DIM-1 -:DIM] = {DIM{1'b1}};
                        new_node_on_one_side_reset[2*DIM-1 -:DIM] = 0;
                        new_node_on_one_side_winner[DIGIT_DIM*DIM*2-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                        new_node_on_one_side_next_node[DIGIT_DIM*DIM*2-1 -:DIGIT_DIM*DIM] = node_list[r_idx];
                        
                    end else begin
                        r_idx = is_in_map(bmu[1], bmu[0]-1);
                        if (r_idx != -1) begin
                            new_node_on_one_side_en[2*DIM-1 -:DIM] = {DIM{1'b1}};
                            new_node_on_one_side_reset[2*DIM-1 -:DIM] = 0;
                            new_node_on_one_side_winner[DIGIT_DIM*DIM*2-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                            new_node_on_one_side_next_node[DIGIT_DIM*DIM*2-1 -:DIGIT_DIM*DIM] = node_list[r_idx];
                            
                        end 
                    end
                end
            end
            spreadable[1] = 1;
        end 
    end
    
    ///////////////////////////////////////////**************************grow_bottom**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (grow_nodes_en && !spreadable[2]) begin
//            $display("new bottom node %d %d", bottomx, bottomy);
            
            new_node_idx_x[2] = bottomx;
            new_node_idx_y[2] = bottomy;
            b_idx = is_in_map(bottomx, bottomy-1);
            if (b_idx != -1) begin
                new_node_in_middle_en[3*DIM-1 -:DIM] = {DIM{1'b1}};
                new_node_in_middle_reset[3*DIM-1 -:DIM] = 0;
                new_node_in_middle_winner[DIGIT_DIM*DIM*3-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                new_node_in_middle_next_node[DIGIT_DIM*DIM*3-1 -:DIGIT_DIM*DIM] = node_list[b_idx];
                
            end else begin
                b_idx = is_in_map(bmu[1], bmu[0]+1);
                if (b_idx != -1) begin
                    new_node_on_one_side_en[3*DIM-1 -:DIM] = {DIM{1'b1}};
                    new_node_on_one_side_reset[3*DIM-1 -:DIM] = 0;
                    new_node_on_one_side_winner[DIGIT_DIM*DIM*3-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                    new_node_on_one_side_next_node[DIGIT_DIM*DIM*3-1 -:DIGIT_DIM*DIM] = node_list[b_idx];
                    
                end else begin
                    b_idx = is_in_map(bmu[1]+1, bmu[0]);
                    if (b_idx != -1) begin
                        new_node_on_one_side_en[3*DIM-1 -:DIM] = {DIM{1'b1}};
                        new_node_on_one_side_reset[3*DIM-1 -:DIM] = 0;
                        new_node_on_one_side_winner[DIGIT_DIM*DIM*3-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                        new_node_on_one_side_next_node[DIGIT_DIM*DIM*3-1 -:DIGIT_DIM*DIM] = node_list[b_idx];
                        
                    end else begin
                        b_idx = is_in_map(bmu[1]-1, bmu[0]);
                        if (b_idx != -1) begin
                            new_node_on_one_side_en[3*DIM-1 -:DIM] = {DIM{1'b1}};
                            new_node_on_one_side_reset[3*DIM-1 -:DIM] = 0;
                            new_node_on_one_side_winner[DIGIT_DIM*DIM*3-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                            new_node_on_one_side_next_node[DIGIT_DIM*DIM*3-1 -:DIGIT_DIM*DIM] = node_list[b_idx];
                        end 
                    end
                end
            end
            spreadable[2] = 1;
        end 
    end
    
    ///////////////////////////////////////////**************************grow_left**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (grow_nodes_en && !spreadable[3]) begin
//            $display("new left node %d %d", leftx, lefty);
            new_node_idx_x[3] = leftx;
            new_node_idx_y[3] = lefty;
            l_idx = is_in_map(leftx-1, lefty);
            if (l_idx != -1) begin
                new_node_in_middle_en[4*DIM-1 -:DIM] = {DIM{1'b1}};
                new_node_in_middle_reset[4*DIM-1 -:DIM] = 0;
                new_node_in_middle_winner[DIGIT_DIM*DIM*4-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                new_node_in_middle_next_node[DIGIT_DIM*DIM*4-1 -:DIGIT_DIM*DIM] = node_list[l_idx];
                
            end else begin
                l_idx = is_in_map(bmu[1]+1, bmu[0]);
                if (l_idx != -1) begin
                    new_node_on_one_side_en[4*DIM-1 -:DIM] = {DIM{1'b1}};
                    new_node_on_one_side_reset[4*DIM-1 -:DIM] = 0;
                    new_node_on_one_side_winner[DIGIT_DIM*DIM*4-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                    new_node_on_one_side_next_node[DIGIT_DIM*DIM*4-1 -:DIGIT_DIM*DIM] = node_list[l_idx];
                    
                end else begin
                    l_idx = is_in_map(bmu[1], bmu[0]+1);
                    if (l_idx != -1) begin
                        new_node_on_one_side_en[4*DIM-1 -:DIM] = {DIM{1'b1}};
                        new_node_on_one_side_reset[4*DIM-1 -:DIM] = 0;
                        new_node_on_one_side_winner[DIGIT_DIM*DIM*4-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                        new_node_on_one_side_next_node[DIGIT_DIM*DIM*4-1 -:DIGIT_DIM*DIM] = node_list[l_idx];
                        
                    end else begin
                        l_idx = is_in_map(bmu[1], bmu[0]-1);
                        if (l_idx != -1) begin
                            new_node_on_one_side_en[4*DIM-1 -:DIM] = {DIM{1'b1}};
                            new_node_on_one_side_reset[4*DIM-1 -:DIM] = 0;
                            new_node_on_one_side_winner[DIGIT_DIM*DIM*4-1 -:DIGIT_DIM*DIM] = node_list[rmu];
                            new_node_on_one_side_next_node[DIGIT_DIM*DIM*4-1 -:DIGIT_DIM*DIM] = node_list[l_idx];
                            
                        end
                    end
                end
            end
            spreadable[3] = 1;
        end
    end
    
    ///////////////////////////////////////////**************************inserted_nodes**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin 
        // insert new node
        for (i=1;i<=4;i=i+1) begin
//            if (new_node_in_middle_en[i*DIM-1 -:DIM]=={4{1'b1}}) $display("in_middle %d", new_node_in_middle_is_done[i*DIM-1 -:DIM]);
//            if (new_node_on_one_side_en[i*DIM-1 -:DIM]=={4{1'b1}}) $display("on_one_side %d", new_node_on_one_side_is_done[i*DIM-1 -:DIM]);
            
            if (new_node_in_middle_is_done[i*DIM-1 -:DIM]=={DIM{1'b1}} && !grow_done[i-1]) begin
                new_node_in_middle_en[i*DIM-1 -:DIM] = 0;
                new_node_in_middle_reset[i*DIM-1 -:DIM] = {DIM{1'b1}};
                insert_new_node(new_node_idx_x[i-1], new_node_idx_y[i-1], new_node_in_middle_weight[DIGIT_DIM*DIM*i-1 -:DIGIT_DIM*DIM]);
                grow_done[i-1] = 1;
//                $display("inserted in_middle %d %h %h", i-1, new_node_idx_x[i-1], new_node_idx_y[i-1]);
            end
            if (new_node_on_one_side_is_done[i*DIM-1 -:DIM]=={DIM{1'b1}} && !grow_done[i-1]) begin
                new_node_on_one_side_en[i*DIM-1 -:DIM] = 0;
                new_node_on_one_side_reset[i*DIM-1 -:DIM] = {DIM{1'b1}};
                insert_new_node(new_node_idx_x[i-1], new_node_idx_y[i-1], new_node_on_one_side_weight[DIGIT_DIM*DIM*i-1 -:DIGIT_DIM*DIM]);
                grow_done[i-1] = 1;
//                $display("inserted on_one_side %d %h %h", i-1, new_node_idx_x[i-1], new_node_idx_y[i-1]);
            end
        end
        if (grow_done==4'b1111 && !node_count_adder_en) begin
            not_man_dist_en = 1;
            grow_nodes_en = 0;
        end
    end
    
    ///////////////////////////////////////////**************************increment_node_count**************************/////////////////////////////////////////////////////
    always @(posedge clk) begin
        if (node_count_adder_en) begin
            nca_num1 = node_count_ieee754;
            nca_num2 = 32'h3F800000; // 1
            nca_en = 1;
            nca_reset = 0;
            if (nca_is_done) begin
                node_count_ieee754 = nca_num_out;
                nca_en = 0;
                nca_reset = 1;
                node_count_adder_en = 0;
            end
        end
    end
    ///////////////////////////////////////////**************************write_weights**************************/////////////////////////////////////////////////////
//    integer file_dir;    
//    always @(posedge clk) begin
//        if (write_en) begin
//            file_dir = $fopen("/home/mad/Documents/Projects/fpga-isom/gsom/weight_out.data", "w");
//            i=0; j=0; k=0;
//            for (i=0; i<=node_count-1; i=i+1) begin
//                for (j=DIM*DIGIT_DIM-1; j>=0; j=j-1) begin
//                    $fwriteb(file_dir, node_list[i][j]);
//                end
//                $fwrite(file_dir, "\n");
////                $display("Written %d %d", i, node_list[i]);
//            end
            
//            #10 $fclose(file_dir);            
//            is_completed = 1;   
//            $finish;
//        end
//    end
    
endmodule