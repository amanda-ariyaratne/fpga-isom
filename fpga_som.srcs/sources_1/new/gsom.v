`timescale 1ns / 1ps

module gsom
    #(
        parameter DIM = 4,
        parameter LOG2_DIM = 3, 
        parameter DIGIT_DIM = 32,
        
        parameter INIT_ROWS = 2,
        parameter INIT_COLS = 2,
        
        parameter MAX_ROWS = 100,
        parameter MAX_COLS = 100,
        
        parameter LOG2_ROWS = 7,         
        parameter LOG2_COLS = 7,

        parameter MAX_NODE_SIZE = 10000,
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
        parameter spread_factor = 32'h3F000000, //0.5
        parameter spread_factor_logval = 32'hBE9A209B, // BE9A209B = -0.30102999566
        
        parameter dimensions_ieee754 = 32'h40800000, // 4
        parameter initial_learning_rate=32'h3E99999A, // 0.3
        parameter smooth_learning_factor= 32'h3F4CCCCD, //0.8
        parameter max_radius=4,
        parameter FD=32'h3DCCCCCD, //0.1
        parameter r=32'h40733333, //3.8
        parameter alpha=32'h3F666666, //0.9
        parameter initial_node_size=30000

    )(
        input wire clk
    );
    
    localparam [DIGIT_DIM-1:0] p_inf = 32'b0111_1111_1111_1111_1111_1111_1111_1111;
    localparam [DIGIT_DIM-1:0] n_inf = 32'b1111_1111_1111_1111_1111_1111_1111_1111;
    
    localparam delta_growing_iter = 34;
    localparam delta_smoothing_iter = 17;
    
    reg signed [LOG2_ROWS:0] i = 0;
    reg signed [LOG2_COLS:0] j = 0;
    reg signed [LOG2_DIM:0] k = 0;
    
    reg [DIGIT_DIM*DIM-1:0] trainX [TRAIN_ROWS-1:0];    
    reg [DIGIT_DIM*DIM-1:0] testX [TEST_ROWS-1:0];
    reg [LOG2_NUM_CLASSES-1:0] trainY [TRAIN_ROWS-1:0];
    reg [LOG2_NUM_CLASSES-1:0] testY [TEST_ROWS-1:0];
    reg [DIGIT_DIM*DIM-1:0] random_weights [INIT_ROWS + INIT_COLS - 1:0];
    
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
    reg [(DIM*DIGIT_DIM)-1:0] node_list [MAX_NODE_SIZE-1:0];
    reg [LOG2_NODE_SIZE-1:0] node_coords [MAX_NODE_SIZE-1:0][1:0];
    reg signed [LOG2_NODE_SIZE*4-1:0] map [MAX_ROWS-1:0][MAX_COLS-1:0];
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
    reg fit_en = 0;
    reg get_LR_en = 0;
    reg grow_en = 0;
    reg adjust_weights_en=0;
    reg check_in_map_en = 0;
    reg smoothing_en=0;
    reg spread_weighs_en=0;
    reg grow_nodes_en=0;
    
    reg dist_enable = 0;
    reg init_neigh_search_en=0;  
    reg nb_search_en=0;
    reg write_en = 0;
    reg is_completed = 0;
    reg min_distance_en=0;
    reg bmu_en=0;
    reg test_mode=0;
    reg classification_en=0;
    
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
        if (init_arrays) begin
            for (i=0; i<MAX_ROWS;i=i+1) begin
                for (j=0; j<MAX_COLS;j=j+1) begin
                    for (k=LOG2_NODE_SIZE; k<=LOG2_NODE_SIZE*4;k=k+LOG2_NODE_SIZE) begin
                        map[i][j][k-1 -:LOG2_NODE_SIZE] <= -1;
                    end
                end    
            end
            
            for (i=0; i<MAX_NODE_SIZE;i=i+1) begin
                node_errors[i] <= 0;  
            end
            init_arrays <= 0;
            init_gsom <= 1;
        end
    end
    
    always @(posedge clk) begin
        if (init_gsom) begin
            map[1][1] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 1;
            node_coords[node_count][1] = 1;
            node_count = node_count + 1;            
            map[1][1][LOG2_NODE_SIZE-1:0] = node_count;
            
            map[1][0] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 1;
            node_coords[node_count][1] = 0;
            node_count = node_count + 1;
            map[1][0][LOG2_NODE_SIZE-1:0] = node_count;
            
            map[0][1] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 0;
            node_coords[node_count][1] = 1;
            node_count = node_count + 1;
            map[0][1][LOG2_NODE_SIZE-1:0] = node_count;
            
            map[0][0] = node_count;
            node_list[node_count] = random_weights[node_count]; // Initialize random weight
            node_coords[node_count][0] = 0;
            node_coords[node_count][1] = 0;
            node_count = node_count + 1;
            map[0][0][LOG2_NODE_SIZE-1:0] = node_count;
            
            node_count_ieee754 = 32'h40800000; // 4
            
            init_gsom = 0;
            init_variables = 1;
        end
        
        if (init_variables) begin
            learning_rate = initial_learning_rate;
            
            // growth threshold            
            mul_num1 = dimensions_ieee754;
            mul_num2 = spread_factor_logval;
            mul_en = 1;
            mul_reset = 0;
            
            init_variables = 0;
        end
        
        if (mul_is_done) begin
            growth_threshold = mul_num_out;
            growth_threshold[DIGIT_DIM-1] = 0; //make the value positive
            mul_en = 0;
            mul_reset = 1;
            fit_en = 1;
        end
    end
    
    always @(posedge clk) begin
        if (fit_en) begin
            current_learning_rate = learning_rate;
            iteration = -1;            
            next_iteration_en = 1;
        end
    end
    
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
    
    always @(posedge clk) begin
        if (next_iteration_en) begin
            if (iteration < GROWING_ITERATIONS) begin
                iteration = iteration + 1;
                // neighbourhood                
                if (iteration <= delta_growing_iter) begin
                    radius = 4;
                end else if ((iteration <= delta_growing_iter*2) && (iteration > delta_growing_iter*1)) begin
                    radius = 2;
                end else if ((iteration <= delta_growing_iter*3) && (iteration > delta_growing_iter*2)) begin
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
            t1 = -1;
            next_t1_en = 1;
        end
    end
    

    
    always @(posedge clk) begin
        if (next_t1_en) begin
            if (t1 < TRAIN_ROWS) begin
                t1 = t1 + 1;   
                dist_enable = 1;        
            end else begin
                next_iteration_en = 1;
            end
            next_t1_en = 0;
        end
    end
    

    
    //////////////////******************************Find BMU******************************/////////////////////////////////
    reg [LOG2_DIM-1:0] iii = 0; 
    
    reg [DIGIT_DIM-1:0] min_distance;   
    reg [LOG2_NODE_SIZE:0] minimum_distance_indices [MAX_NODE_SIZE-1:0][1:0];
    reg [LOG2_NODE_SIZE:0] minimum_distance_1D_indices [MAX_NODE_SIZE:0];
    reg [LOG2_DIM-1:0] min_distance_next_index = 0;
    
    reg [LOG2_ROWS:0] idx_i;
    reg [LOG2_COLS:0] idx_j;   
    
    wire [DIGIT_DIM-1:0] distance_out [MAX_NODE_SIZE-1:0];
    wire [MAX_NODE_SIZE-1:0] distance_done;
    reg distance_en=0;
    reg distance_reset=0;
    reg [DIGIT_DIM*DIM-1:0] distance_X=0;
    
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
                .index(euc_i),
                .num_out(distance_out[euc_i]),
                .is_done(distance_done[euc_i])
            );
        end
    endgenerate
    
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
    
    always @(posedge clk) begin
        if (dist_enable) begin
            distance_X=trainX[t1];
            distance_reset=0;
            distance_en=1;
            
            if (distance_done == {MAX_NODE_SIZE{1'b1}}) begin
                distance_en = 0;
                distance_reset = 1;
                i = 0;
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
            comp_in_2 = distance_out[i];
            comp_reset = 0;
            comp_en = 1;
            
            if (comp_done) begin
                comp_en = 0;
                comp_reset = 1;
                
                if (comp_out==0) begin
                    minimum_distance_indices[min_distance_next_index][1] = node_coords[i][1];
                    minimum_distance_indices[min_distance_next_index][0] = node_coords[i][0];
                    minimum_distance_1D_indices[min_distance_next_index] = i;
                    min_distance_next_index = min_distance_next_index + 1;
                
                end else if (comp_out==1) begin
                    min_distance = distance_out[i];
                    minimum_distance_indices[0][1] = node_coords[i][1];
                    minimum_distance_indices[0][0] = node_coords[i][0]; 
                    minimum_distance_1D_indices[0] = i;                   
                    min_distance_next_index = 1;
                end 
                
                i=i+1;
                if (i>=node_count) begin
                    min_distance_en=0;
                    bmu_en=1;
                end
            end
        end
    end
    
    reg [LOG2_NODE_SIZE:0] bmu [1:0];
    reg [LOG2_NODE_SIZE:0] rmu;
    
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
    
    //////////////////////************Start Neighbourhood search************//////////////////////////////////////////
    
    reg signed [LOG2_ROWS+1:0] bmu_i;
    reg signed [LOG2_COLS+1:0] bmu_j;
    
    reg signed [LOG2_ROWS+1:0] abs_bmu_i;
    reg signed [LOG2_COLS+1:0] abs_bmu_j;
    
    reg [1:0] i_j_signs;
    
    reg signed [LOG2_ROWS+1:0] rmu_i;
    
    reg signed [LOG2_ROWS+1:0] bmu_x;
    reg signed [LOG2_COLS+1:0] bmu_y;
    reg signed [DIGIT_DIM-1 :0] man_dist; /////////// not sure
    
   
    always @(posedge clk) begin    
        if (init_neigh_search_en) begin
            bmu_x = bmu[1]; bmu_y = bmu[0];  
            bmu_i = bmu_x-radius;            
            bmu_j = bmu_y-radius;
            init_neigh_search_en=0;
            nb_search_en=1;
        end
    end
    
    integer digit;
    
    reg update_en=0;
    reg update_reset=0;
    reg [DIGIT_DIM*DIM-1:0] update_in_1;
    reg [DIGIT_DIM*DIM-1:0] update_in_2;
    reg [DIGIT_DIM-1:0]  update_learning_rate;
    wire [DIGIT_DIM*DIM-1:0] update_out;
    wire [DIGIT_DIM*DIM-1:0] update_error_out;
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
                .error(update_error_out[update_i*DIGIT_DIM-1 -:DIGIT_DIM]),
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
    
    reg not_man_dist_en = 0;

    always @(posedge clk) begin    
        if (nb_search_en && !update_en && !update_neighbour_en) begin  
            man_dist = (bmu_x-bmu_i) >= 0 ? (bmu_x-bmu_i) : (bmu_i-bmu_x);
            man_dist = man_dist + ((bmu_y - bmu_j)>= 0 ? (bmu_y - bmu_j) : (bmu_j - bmu_y));              
            if (man_dist == 0) begin
                update_in_1 = node_list[rmu];
                update_in_2 = trainX[t1];
                update_learning_rate = current_learning_rate;
                update_en=1; 
                update_reset=0;             
            end else if (man_dist <= radius) begin
                update_neighbour_in_1 = node_list[rmu];
                update_neighbour_in_2 = node_list[rmu_i];
                update_neighbour_learning_rate = current_learning_rate;
                update_neighbour_man_dist = man_dist;
                update_neighbour_en=1; 
                update_neighbour_reset=0;             
            end else begin
                not_man_dist_en = 1;
                nb_search_en = 0;
            end
        end
    end
    
    reg signed [LOG2_NODE_SIZE:0] leftx, lefty, rightx, righty, upx, upy, bottomx, bottomy;
    reg signed [LOG2_NODE_SIZE:0] leftx_abs, lefty_abs, rightx_abs, righty_abs, upx_abs, upy_abs, bottomx_abs, bottomy_abs;
    
    reg [LOG2_NODE_SIZE:0] x, y;
    reg [LOG2_NODE_SIZE:0] abs_x, abs_y;
    reg [1:0] x_y_signs;
    reg is_exist=0;
    reg check_in_map_done=0;
    
    
    always @(posedge clk) begin
        if (check_in_map_en) begin
            check_in_map_done=0;
            abs_x = x>0 ? x : -x;
            abs_y = y>0 ? y : -y;
            
            x_y_signs[0] = x>0 ? 0 : 1;
            x_y_signs[1] = y>0 ? 0 : 1;
            
            if ((map[abs_x][abs_y][LOG2_NODE_SIZE*1-1 -:LOG2_NODE_SIZE] != -1) && (x_y_signs[0]==0) && (x_y_signs[1]==0)) begin
                is_exist = 1;
            end else if ((map[abs_x][abs_y][LOG2_NODE_SIZE*2-1 -:LOG2_NODE_SIZE] != -1) && (x_y_signs[0]==0) && (x_y_signs[1]==1)) begin
                is_exist = 1;
            end else if ((map[abs_x][abs_y][LOG2_NODE_SIZE*3-1 -:LOG2_NODE_SIZE] != -1) && (x_y_signs[0]==1) && (x_y_signs[1]==0)) begin
                is_exist = 1;
            end else if ((map[abs_x][abs_y][LOG2_NODE_SIZE*4-1 -:LOG2_NODE_SIZE] != -1) && (x_y_signs[0]==1) && (x_y_signs[1]==1)) begin
                is_exist = 1;
            end else begin
                is_exist = 0;
            end
            check_in_map_en = 0;
            check_in_map_done = 1;
        end
    end
    
    reg [3:0] spreadable;
    reg [1:0] is_spreadable_finished_idx;
    

    always @(posedge clk) begin
        if (adjust_weights_en) begin
            comp_reset = 0;
            comp_en = 1;
            comp_in_1 = node_errors[rmu];
            comp_in_2 = growth_threshold;
            
            adjust_weights_en = 0;
        end
        if (comp_done) begin
            comp_reset = 1;
            comp_en = 0;
            if (comp_out == 1) begin
                leftx = bmu_x-1;    lefty = bmu_y;
                rightx = bmu_x+1;   righty = bmu_y;
                upx = bmu_x;        upy = bmu_y+1;
                bottomx = bmu_x;    bottomy = bmu_y-1;
                
                x = upx; y=upy;                
                check_in_map_en = 1;
                is_spreadable_finished_idx = 0;                                
            
            end else begin
                smoothing_en=1;
            end
        end
        
        if (is_spreadable_finished_idx == 3) begin // checked all neighbours exist
            if (spreadable == 4'b1) begin
                spread_weighs_en=1;
            end else begin
                grow_nodes_en=1;
            end
            is_spreadable_finished_idx = 0;            
        end
        
        if (check_in_map_done) begin
            spreadable[is_spreadable_finished_idx] = is_exist;
            
            check_in_map_en = 1;
            check_in_map_done = 0;

            if (is_spreadable_finished_idx==0) begin
                x = rightx; y=righty;
                is_spreadable_finished_idx = is_spreadable_finished_idx+1;
                
            end else if (is_spreadable_finished_idx == 1) begin
                x = bottomx; y=bottomy;
                is_spreadable_finished_idx = is_spreadable_finished_idx+1;
                
            end else if (is_spreadable_finished_idx == 2) begin
                x = leftx; y=lefty;
                is_spreadable_finished_idx = is_spreadable_finished_idx+1;
                
            end else begin
                check_in_map_en = 0;
            end 
        end
    end
    
    always @(posedge clk) begin
        if ((update_done == {DIM{1'b1}}) || (update_neighbour_done == {DIM{1'b1}}) || not_man_dist_en) begin
            if (update_done == {DIM{1'b1}}) begin
                node_list[rmu] = update_out;
                update_en=0;
                update_reset=1;
                node_errors[rmu] = node_errors[rmu] + update_error_out;
                adjust_weights_en = 1;
            end 
            
            if (update_neighbour_done == {DIM{1'b1}}) begin
                node_list[rmu_i] = update_neighbour_out;
                update_neighbour_en=0;
                update_neighbour_reset=1;
            end          
                
            bmu_j = bmu_j + 1;
            
            if (bmu_j < bmu_y+radius+1) begin
                bmu_j = bmu_y-radius;                
                bmu_i = bmu_i + 1;
                
                if (bmu_i < bmu_x+radius+1) begin
                    nb_search_en = 0; // neighbourhood search finished        
                    next_t1_en = 1; // go to the next input
                end else
                    nb_search_en = 1; // next neighbour
                
                not_man_dist_en = 0; // close this block
                
            end else begin
                nb_search_en = 1; // next neighbour
                not_man_dist_en = 0; // close this block
            end
            
            if (nb_search_en) begin
                abs_bmu_i = bmu_i>0 ? bmu_i : -bmu_i;
                abs_bmu_j = bmu_j>0 ? bmu_j : -bmu_j;
                
                i_j_signs[0] = bmu_i>0 ? 0 : 1;
                i_j_signs[1] = bmu_j>0 ? 0 : 1;
                
                if ((map[abs_bmu_i][abs_bmu_j][LOG2_NODE_SIZE*1-1 -:LOG2_NODE_SIZE] != -1) && (i_j_signs[0]==0) && (i_j_signs[1]==0)) begin
                    rmu_i = map[abs_bmu_i][abs_bmu_j][LOG2_NODE_SIZE*1-1 -:LOG2_NODE_SIZE];
                    nb_search_en = 1;
                end else if ((map[abs_bmu_i][abs_bmu_j][LOG2_NODE_SIZE*2-1 -:LOG2_NODE_SIZE] != -1) && (i_j_signs[0]==0) && (i_j_signs[1]==1)) begin
                    rmu_i = map[abs_bmu_i][abs_bmu_j][LOG2_NODE_SIZE*2-1 -:LOG2_NODE_SIZE];
                    nb_search_en = 1;
                end else if ((map[abs_bmu_i][abs_bmu_j][LOG2_NODE_SIZE*3-1 -:LOG2_NODE_SIZE] != -1) && (i_j_signs[0]==1) && (i_j_signs[1]==0)) begin
                    rmu_i = map[abs_bmu_i][abs_bmu_j][LOG2_NODE_SIZE*3-1 -:LOG2_NODE_SIZE];
                    nb_search_en = 1;
                end else if ((map[abs_bmu_i][abs_bmu_j][LOG2_NODE_SIZE*4-1 -:LOG2_NODE_SIZE] != -1) && (i_j_signs[0]==1) && (i_j_signs[1]==1)) begin
                    rmu_i = map[abs_bmu_i][abs_bmu_j][LOG2_NODE_SIZE*4-1 -:LOG2_NODE_SIZE];
                    nb_search_en = 1;
                end else begin
                    nb_search_en = 0;
                    not_man_dist_en = 1;
                end
            end
        end
    end
    
endmodule
