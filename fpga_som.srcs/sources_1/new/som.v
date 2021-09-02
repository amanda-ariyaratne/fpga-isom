`timescale 1ns / 1ps

module som
    #(
        parameter DIM = 4,
        parameter LOG2_DIM = 3, 
        parameter DIGIT_DIM = 32,
        
        parameter ROWS = 10,
        parameter LOG2_ROWS = 4, 
        parameter COLS = 10,
        parameter LOG2_COLS = 4,     
        
        parameter TRAIN_ROWS = 75,
        parameter LOG2_TRAIN_ROWS = 7,
        parameter TEST_ROWS = 150,
        parameter LOG2_TEST_ROWS = 8, 
        
        parameter NUM_CLASSES = 3+1,
        parameter LOG2_NUM_CLASSES = 2, 
        
        parameter TOTAL_ITERATIONS=4,              
        parameter LOG2_TOT_ITERATIONS = 3,
        
        parameter INITIAL_NB_RADIUS = 3,
        parameter NB_RADIUS_STEP = 1,
        parameter LOG2_NB_RADIUS = 2,
        parameter ITERATION_NB_STEP = 1,
        
        parameter INITIAL_ALPHA = 32'b00111111011001100110011001100110, //0.9
        parameter ALPHA_STEP = 32'b00111101110011001100110011001101,//0.1
        // 0.9 = 32'b00111111011001100110011001100110
        // 0.1 - 32'b00111101110011001100110011001101
        // 0.2 - 32'b00111110010011001100110011001101
        // 0.05 - 32'b00111101010011001100110011001101
        
        parameter LOG2_ALPHA = 32,
        parameter ITERATION_STEP = 1,          
        parameter STEP = 4,
        
        parameter RAND_NUM_BIT_LEN = 10        
    )
    (
        input wire clk,
        output wire [LOG2_TEST_ROWS:0] prediction,
        output wire completed
    );

    ///////////////////////////////////////////////////////*******************Declare enables***********/////////////////////////////////////
    reg [1:0] training_en = 0;
    reg [1:0] next_iteration_en=0;
    reg [1:0] next_x_en=0;    
    reg [1:0] dist_enable = 0;
    reg [1:0] init_neigh_search_en=0;  
    reg [1:0] nb_search_en=0;
    reg [1:0] test_en = 0;
    reg [1:0] classify_x_en = 0;
    reg [1:0] classify_weights_en = 0;
    reg [1:0] init_classification_en=0;
    reg [1:0] classification_en = 0;
    reg [1:0] class_label_en=0;
    reg write_en = 0;
    reg is_completed = 0;
    reg iterate_en=0;
    reg bmu_en=0;
    reg classify_init_en=0;
    reg test_mode=0;
    reg update_alpha_en=0;
    
    ///////////////////////////////////////////////////////*******************Other variables***********/////////////////////////////////////
    
    reg signed [LOG2_TOT_ITERATIONS:0] iteration;
    reg signed [LOG2_NB_RADIUS:0] nb_radius = INITIAL_NB_RADIUS;
    reg signed [LOG2_ALPHA-1:0] alpha = INITIAL_ALPHA;  
    
    reg [LOG2_ROWS:0] ii = 0;
    reg [LOG2_COLS:0] jj = 0;
    reg [LOG2_NUM_CLASSES:0] kk = 0;
    
    reg [LOG2_COLS:0] bmu [1:0];
    reg [LOG2_TRAIN_ROWS:0] class_frequency_list [ROWS-1:0][COLS-1:0][NUM_CLASSES-1:0];
    
    localparam [DIGIT_DIM-1:0] p_inf = 32'b0111_1111_1111_1111_1111_1111_1111_1111;
    localparam [DIGIT_DIM-1:0] n_inf = 32'b1111_1111_1111_1111_1111_1111_1111_1111;
    
    ///////////////////////////////////////////////////////*******************File read variables***********/////////////////////////////////////
    
    
    reg [DIGIT_DIM-1:0] weights [ROWS-1:0][COLS-1:0][DIM-1:0];
    reg [DIGIT_DIM-1:0] trainX [TRAIN_ROWS-1:0][DIM-1:0];    
    reg [DIGIT_DIM-1:0] testX [TEST_ROWS-1:0][DIM-1:0];
    reg [LOG2_NUM_CLASSES-1:0] trainY [TRAIN_ROWS-1:0];
    reg [LOG2_NUM_CLASSES-1:0] testY [TEST_ROWS-1:0];
    
    reg signed [LOG2_ROWS:0] i = 0;
    reg signed [LOG2_COLS:0] j = 0;
    reg signed [LOG2_DIM:0] k = DIM-1;
    reg signed [LOG2_DIM:0] kw = DIM-1;
    reg signed [LOG2_DIM:0] k1 = DIM-1;
    reg signed [LOG2_DIM:0] k2 = DIM-1;    
    
    reg signed [LOG2_TRAIN_ROWS:0] t1 = 0;
    reg signed [LOG2_TEST_ROWS:0] t2 = 0;
    
    integer weights_file;
    integer trains_file;
    integer test_file;
    
    reg [(DIM*DIGIT_DIM)-1:0] rand_v;
    reg [(DIM*DIGIT_DIM)+LOG2_NUM_CLASSES-1:0] temp_train_v;
    reg [(DIM*DIGIT_DIM)+LOG2_NUM_CLASSES-1:0] temp_test_v;
    
    integer eof_weight;
    integer eof_train;
    integer eof_test;
    
    ///////////////////////////////////////////////////////*******************Read weight vectors***********/////////////////////////////////////
    initial begin
        weights_file = $fopen("/home/aari/Projects/Vivado/fpga_som/som/weights.data","r");
        while (!$feof(weights_file))
        begin
            eof_weight = $fscanf(weights_file, "%b\n",rand_v);
            
            for(kw=DIM;kw>0;kw=kw-1) begin
                weights[i][j][kw-1] = rand_v[(DIGIT_DIM*kw)+1-:DIGIT_DIM];
            end
            
            j = j + 1;
            if (j == COLS)
            begin
                j = 0;
                i = i + 1;
            end
        end
        $fclose(weights_file);
    end
    
    ///////////////////////////////////////////////////////*******************Read train vectors***********/////////////////////////////////////
    initial begin
        trains_file = $fopen("/home/aari/Projects/Vivado/fpga_som/som/train.data","r");
        while (!$feof(trains_file)) begin        
            eof_train = $fscanf(trains_file, "%b\n",temp_train_v);
            
            for(k1=DIM;k1>0;k1=k1-1) begin
                trainX[t1][k1-1] = temp_train_v[(DIGIT_DIM*k1)+LOG2_NUM_CLASSES-1 -:DIGIT_DIM];
            end
            trainY[t1] = temp_train_v[LOG2_NUM_CLASSES-1:0];
            t1 = t1 + 1;
        end
        $fclose(trains_file);
        training_en = 1;
    end

    ///////////////////////////////////////////////////////*******************Read test vectors***********/////////////////////////////////////
    initial begin
        test_file = $fopen("/home/aari/Projects/Vivado/fpga_som/som/test.data","r");
        while (!$feof(test_file))
        begin
            eof_test = $fscanf(test_file, "%b\n",temp_test_v);
            for(k2=DIM;k2>0;k2=k2-1)
            begin
                testX[t2][k2-1] = temp_test_v[(DIGIT_DIM*k1)+LOG2_NUM_CLASSES-1 -:DIGIT_DIM];
            end
                
            testY[t2] = temp_test_v[LOG2_NUM_CLASSES-1:0];
            t2 = t2 + 1;
        end
        $fclose(test_file);  
    end
    
    ////////////////////*****************************Initialize frequenct list*************//////////////////////////////
    initial
    begin
        for (ii = 0; ii < ROWS; ii = ii + 1)
        begin
            for (jj = 0; jj < COLS; jj = jj + 1)
            begin
                for (kk = 0; kk < NUM_CLASSES; kk = kk + 1)
                begin
                    class_frequency_list[ii][jj][kk] = 0;
                end
            end
        end
        $display("class frequnecy list initialized");
    end
    
    ///////////////////////////////////////////////////////****************Start LFSR**************/////////////////////////////////////
    
    reg lfsr_en = 1;
    reg seed_en = 1;
    wire [(DIM*RAND_NUM_BIT_LEN)-1:0] random_number_arr;
    
    genvar dim_i;
    
    generate
        for(dim_i=1; dim_i <= DIM; dim_i=dim_i+1)
        begin
            lfsr #(.NUM_BITS(RAND_NUM_BIT_LEN)) lfsr_rand
            (
                .i_Clk(clk),
                .i_Enable(lfsr_en),
                .i_Seed_DV(seed_en),
                .i_Seed_Data(dim_i[RAND_NUM_BIT_LEN-1:0]),
                .o_LFSR_Data(random_number_arr[(dim_i*RAND_NUM_BIT_LEN)-1 : (dim_i-1)*RAND_NUM_BIT_LEN])
            );
        end
    endgenerate
    
    ///////////////////////////////////////////////////////*******************Start Training***********/////////////////////////////////////
    always @(posedge clk)
    begin
        if (training_en)
        begin
            $display("training_en");
            iteration = -1;
            next_iteration_en = 1;
            training_en = 0;
        end
    end
    
    integer step_i;
    always @(posedge clk)
    begin
        if (next_iteration_en)
        begin
            t1 = -1; // reset trainset pointer
            if (iteration<(TOTAL_ITERATIONS-1)) begin
                // change current iteration
                iteration = iteration + 1;
                $display("iteration ", iteration);
                
                // change update alpha
                for (step_i=1; step_i<=STEP;step_i = step_i+1) begin
                    if ((iteration==(ITERATION_STEP*step_i))) begin
                        update_alpha_en=1;
                    end
                end
                // change neighbouhood radius
                for (step_i=1; step_i<=4;step_i = step_i+1) begin
                    if ( (iteration==(ITERATION_NB_STEP*step_i) ) ) begin
                        nb_radius <=  nb_radius-NB_RADIUS_STEP;
                    end
                end
                
                next_x_en = 1;                
            end
            else begin
                iteration = -1;                
                next_x_en = 0;
                init_classification_en = 1; // start classification
            end
            
            next_iteration_en = 0;            
        end
    end
    
    reg alpha_reset=0;
    reg alpha_en=0;
    reg [DIGIT_DIM-1:0] alpha_in_1;
    reg [DIGIT_DIM-1:0] alpha_in_2;
    wire [DIGIT_DIM-1:0] alpha_out;
    wire alpha_done;
    
    fpa_adder alpha_update(
        .clk(clk),
        .reset(alpha_reset),
        .en(alpha_en),
        .num1(alpha_in_1),
        .num2(alpha_in_2),
        .num_out(alpha_out),
        .is_done(alpha_done)
    );
    
    always @(posedge clk) begin
        if (update_alpha_en) begin
            alpha_in_1=alpha;
            alpha_in_2=ALPHA_STEP;
            alpha_en=1;
            
            if (alpha_done) begin
                alpha = alpha_out;
                alpha_en=0;
                alpha_reset=1;
                update_alpha_en=0;
            end
        end
    end
    
    always @(posedge clk)
    begin
        if (next_x_en && !classification_en)
        begin                
            if (t1<TRAIN_ROWS-1)
            begin        
                t1 = t1 + 1;
                $display("t1 ", t1);
                dist_enable = 1;
            end            
            else
            begin
                $display("next_iteration_en ", iteration); 
                next_iteration_en = 1;  
            end
                               
            next_x_en = 0;
        end
    end
    
    /////////////////////////////////////******************************Classification logic******************************/////////////////////////////////
    always @(posedge clk)
    begin
        if (init_classification_en)
        begin
            $display("init_classification_en"); 
            lfsr_en = 0; // turn off the random number generator
            next_x_en = 1;
            classification_en = 1;
            init_classification_en = 0;
        end
    end
    
    always @(posedge clk)
    begin
        if (next_x_en && classification_en)
        begin       
            // classify prev x 's bmu
            if (t1>=0)
                class_frequency_list[bmu[1]][bmu[0]][trainY[t1]] =  class_frequency_list[bmu[1]][bmu[0]][trainY[t1]] + 1;
                      
            if (t1<TRAIN_ROWS-1)
            begin                           
                t1 = t1 + 1;
                dist_enable = 1;
                $display("classify ", t1);    
            end            
            else
            begin    
                $display("classification_en STOPPED"); 
                classification_en = 0;          
                class_label_en = 1;                
            end 
                         
            next_x_en = 0;
        end
    end
    
    //////////////////******************************Find BMU******************************/////////////////////////////////
    reg [LOG2_DIM-1:0] iii = 0; 
    
    reg [LOG2_DIM:0] hamming_distance;
    reg [DIGIT_DIM-1:0] min_distance;   
    reg [LOG2_COLS:0] minimum_distance_indices [(ROWS*COLS-1):0][1:0];
    reg [LOG2_DIM-1:0] min_distance_next_index = 0;
    
    reg [LOG2_DIM:0] hash_count;    
    reg [LOG2_DIM:0] min_hash_count;
    reg [LOG2_DIM:0] hash_counts [ROWS-1:0][COLS-1:0]; 
        
    reg [LOG2_ROWS:0] idx_i;
    reg [LOG2_COLS:0] idx_j;   
    
    reg [DIGIT_DIM-1:0] w;      
    reg [DIGIT_DIM-1:0] x;  
    
    reg [DIGIT_DIM*DIM-1:0] distance_in_1;
    reg [DIGIT_DIM*DIM-1:0] distance_in_2;
    wire [DIGIT_DIM-1:0] distance_out;
    wire distance_done=0;
    reg distance_en=0;
    reg distance_reset;
    
    integer signed assign_i;
    integer signed assign_j;
    reg [DIGIT_DIM-1:0] comp_in_1;
    reg [DIGIT_DIM-1:0] comp_in_2;
    wire [1:0] comp_out;
    wire comp_done=0;
    reg comp_en=0;
    reg comp_reset=1;
    
    fpa_euclidean_distance euclidean_distance(
        .clk(clk),
        .en(distance_en),
        .reset(distance_reset),
        .weight(distance_in_1),
        .trainX(distance_in_2),
        .num_out(distance_out),
        .is_done(distance_done)
    );   
    
    fpa_comparator get_max(
        .clk(clk),
        .en(comp_en),
        .reset(comp_reset),
        .num1(comp_in_1),
        .num2(comp_in_2),
        .num_out(comp_out),
        .is_done(comp_done)
    );
    
    always @(posedge clk)
    begin
        if (dist_enable) begin
            i = 0;
            j = 0;
            min_distance_next_index = 0;
            min_distance = p_inf;
            dist_enable=0;
            iterate_en=1;
        end
    end
    
    always @(posedge clk) begin
        if (iterate_en) begin
            for(assign_i=DIM-1; assign_i>0; assign_i=assign_i-1) begin
                for(assign_j=DIGIT_DIM-1; assign_j>0; assign_j=assign_j-1) begin
                    distance_in_1[(assign_i*DIM)+assign_j] = weights[i][j][assign_i][assign_j];
                    distance_in_2[(assign_i*DIM)+assign_j] = trainX[t1][assign_i][assign_j];
                end                
            end
            
            distance_en=1;            
            iterate_en=0;
        end
    end
    
    always @(posedge clk) begin
        if (distance_done) begin
            distance_en=0;
            $display("Calculated");
            comp_in_1 = min_distance;
            comp_in_2 = distance_out;
            comp_en = 1;
            
            if (comp_done) begin
                comp_en=0;
                
                distance_reset=1;
                comp_reset=1;
                
                if (comp_out==0) begin
                    minimum_distance_indices[min_distance_next_index][1] = i;
                    minimum_distance_indices[min_distance_next_index][0] = j;
                    min_distance_next_index = min_distance_next_index + 1;
                
                end else if (comp_out==1) begin
                    min_distance = distance_out;
                    minimum_distance_indices[0][1] = i;
                    minimum_distance_indices[0][0] = j;                        
                    min_distance_next_index = 1;
                end
                
                if (j==COLS-1) begin
                    j=0;
                    i=i+1;
                end else begin
                    j=j+1;
                end
                
                if (i==ROWS-1) begin
                    bmu_en=1;
                end else begin
                    for(assign_i=0; assign_i<DIGIT_DIM; assign_i=assign_i+1) begin
                        distance_in_1[assign_i] = weights[i][j][assign_i];
                        distance_in_2[assign_i] = trainX[t1][assign_i];
                    end
                    distance_en=1;
                end
            end
        end
    end
    
    always @(posedge clk) begin
        if (bmu_en) begin
            bmu[1] = minimum_distance_indices[0][1];
            bmu[0] = minimum_distance_indices[0][0];
            
            if (!classification_en)
                init_neigh_search_en = 1;
            else
                next_x_en = 1;
            bmu_en=0;
        end
    end
    
    //////////////////////************Start Neighbourhood search************//////////////////////////////////////////
    
    reg signed [LOG2_ROWS+1:0] bmu_i;
    reg signed [LOG2_COLS+1:0] bmu_j;
    reg signed [LOG2_ROWS+1:0] bmu_x;
    reg signed [LOG2_COLS+1:0] bmu_y;
    reg signed [LOG2_NB_RADIUS+1:0] man_dist; /////////// not sure
    
   
    always @(posedge clk)
    begin    
        if (init_neigh_search_en) begin
            bmu_x = bmu[1]; bmu_y = bmu[0];  
            bmu_i = (bmu_x-nb_radius) < 0 ? 0 : (bmu_x-nb_radius);            
            bmu_j = (bmu_y-nb_radius) < 0 ? 0 : (bmu_y-nb_radius);
            init_neigh_search_en=0;
            nb_search_en=1;
        end
    end
    
    integer digit;
    
    reg update_en=0;
    reg update_reset=0;
    reg [DIGIT_DIM-1:0] update_in_1;
    reg [DIGIT_DIM-1:0] update_in_2;
    wire [DIGIT_DIM-1:0] update_out;
    wire update_done=0;
    
    fpa_update_weight update(
        .clk(clk),
        .en(update_en),
        .reset(update_reset),
        .weight(update_in_1),
        .train_row(alpha),
        .alpha(update_in_2),
        .num_out(update_out),
        .is_done(update_done)
    );

    always @(posedge clk)
    begin    
        if (nb_search_en && !update_en) begin  
            man_dist = (bmu_x-bmu_i) >= 0 ? (bmu_x-bmu_i) : (bmu_i-bmu_x);
            man_dist = man_dist + ((bmu_y - bmu_j)>= 0 ? (bmu_y - bmu_j) : (bmu_j - bmu_y));              
            
            if (man_dist <= nb_radius) begin
                
                for (digit=0; digit<DIM; digit=digit+1) begin
                   update_in_1[digit] = weights[bmu_i][bmu_j][digit];
                   update_in_2[digit] = trainX[t1][digit];
                end  
                update_en=1;              
            end
        end
    end
    
    always @(posedge clk) begin
        if (update_done) begin
            update_en=0;
            update_reset=1;
            
            bmu_j = bmu_j + 1;
            
            if (bmu_j == bmu_y+nb_radius+1 || bmu_j == COLS) begin
                bmu_j = (bmu_y-nb_radius) < 0 ? 0 : (bmu_y-nb_radius);                
                bmu_i = bmu_i + 1;
            end            
            if (bmu_i == bmu_x+nb_radius+1 || bmu_i==ROWS) begin
                nb_search_en = 0; // neighbourhood search finished        
                next_x_en = 1; // go to the next input
            end
        end
    end
    
    /////////////////////************Start Classification of weight vectors********///////////////////////
    reg [LOG2_NUM_CLASSES:0] class_labels [ROWS-1:0][COLS-1:0];    

    integer most_freq = 0;
    reg [3:0] default_freq [NUM_CLASSES-1:0];
    
    always @(posedge clk)
    begin
        if (class_label_en)
        begin
            i=0;j=0;k=0;
            for(i=0;i<ROWS;i=i+1)
            begin
                for(j=0;j<COLS;j=j+1)
                begin
                    most_freq = 0;
                    class_labels[i][j] = NUM_CLASSES-1; /////////// hardcoded default value
                    for(k=0;k<NUM_CLASSES-1;k=k+1)
                    begin
                        if (class_frequency_list[i][j][k]>most_freq)
                        begin
                            class_labels[i][j] = k;
                            most_freq = class_frequency_list[i][j][k];
                        end
                    end
                    if (class_labels[i][j] == NUM_CLASSES) /////////// hardcoded default value
                    begin                        
                        // reset array
                        for(k=0;k<=NUM_CLASSES-1;k=k+1)
                        begin
                            default_freq[k] = 0;
                        end
                        
                        if (i-1>0)
                        begin
                            k = class_labels[i-1][j];
                            default_freq[k] = default_freq[k] +1;
                        end
                        
                        if (i+1<ROWS)
                        begin
                            k = class_labels[i+1][j];
                            default_freq[k] = default_freq[k] +1;
                        end
                        
                        if (j-1>0)
                        begin
                            k = class_labels[i][j-1];
                            default_freq[k] = default_freq[k] +1;
                        end
                        
                        if (j+1<COLS)
                        begin
                            k = class_labels[i][j+1];
                            default_freq[k] = default_freq[k] +1;
                        end
                        
                        most_freq = 0;
                        for(k=0;k<=NUM_CLASSES-2;k=k+1) // only check 0,1,2
                        begin
                            if (default_freq[k] >= most_freq)
                            begin
                                most_freq = default_freq[k];
                                class_labels[i][j] = k;
                            end
                        end                      
                    end
                end
            end
            class_label_en = 0;
            test_en = 1;
            test_mode=1;
            t2 = -1;
        end
    end
    
    //////////////////////////////***************Start test************************///////////////////////////////////////////////////////
    
    always @(posedge clk)
    begin
        if (test_en)
        begin
            if (t2<TEST_ROWS-1)
            begin
                t2 = t2 + 1;                
                classify_x_en = 1;
            end            
            else
            begin 
                test_mode=0;
                test_en = 0;
                write_en = 1;            
            end
        end
    end
    
    reg [LOG2_TEST_ROWS:0] correct_predictions = 0; // should take log2 of test rows
    reg [LOG2_NUM_CLASSES:0] predictionY [TEST_ROWS-1:0];
    reg [LOG2_TEST_ROWS:0] tot_predictions = 0;
    
    always @(posedge clk) begin
        if (classify_x_en) begin
            i = 0;
            j = 0;
            min_distance_next_index = 0;
            min_distance = p_inf;
            classify_x_en=0;
            classify_init_en=1;
        end
    end
    
    always @(posedge clk) begin
        if (classify_init_en) begin
            distance_reset=1; 
            comp_reset=1; 
            for(assign_i=0; assign_i<DIGIT_DIM; assign_i=assign_i+1) begin
                distance_in_1[assign_i] = weights[i][j][assign_i];
                distance_in_2[assign_i] = testX[t1][assign_i];
            end
            distance_en=1;            
            classify_init_en=0;
        end
    end
    
    always @(posedge clk) begin
        if (distance_done && test_mode) begin
            distance_en=0;

            comp_in_1 = min_distance;
            comp_in_2 = distance_out;
            comp_en = 1;
            
            if (comp_done) begin
                comp_en=0;
                
                distance_reset=1;
                comp_reset=1;
                
                if (comp_out==0) begin
                    minimum_distance_indices[min_distance_next_index][1] = i;
                    minimum_distance_indices[min_distance_next_index][0] = j;
                    min_distance_next_index = min_distance_next_index + 1;
                
                end else if (comp_out==1) begin
                    min_distance = distance_out;
                    minimum_distance_indices[0][1] = i;
                    minimum_distance_indices[0][0] = j;                        
                    min_distance_next_index = 1;
                end
                
                if (j==COLS-1) begin
                    j=0;
                    i=i+1;
                end else begin
                    j=j+1;
                end
                
                if (i==ROWS-1) begin
                    bmu_en=1;
                end else begin
                    for(assign_i=0; assign_i<DIGIT_DIM; assign_i=assign_i+1) begin
                        distance_in_1[assign_i] = weights[i][j][assign_i];
                        distance_in_2[assign_i] = testX[t1][assign_i];
                    end
                    distance_en=1;
                end
            end
        end
    end
    
    always @(posedge clk) begin
        if (bmu_en && test_mode) begin
            bmu[1] = minimum_distance_indices[0][1];
            bmu[0] = minimum_distance_indices[0][0];
            
            if (class_labels[bmu[1]][bmu[0]] == testY[t2])
            begin
                correct_predictions = correct_predictions + 1;                
            end
            
            predictionY[t2] = class_labels[bmu[1]][bmu[0]];  
            bmu_en = 0;   
            test_en = 1;
        end
    end
    
    integer fd;    
    always @(posedge clk) begin
        if (write_en) begin
            fd = $fopen("/home/mad/Documents/fpga-isom/som/weight_out.data", "w");
            i=0; j=0; k=0;
            for (i=0; i<=ROWS-1; i=i+1) begin
                for (j=0; j<=COLS-1; j=j+1) begin
                    for (k=DIM-1; k>=0; k=k-1) begin                        
                        $fwriteb(fd, weights[i][j][k]);
                    end
                    $fwrite(fd, "\n");
                end
            end
            
            #10 $fclose(fd);            
            is_completed = 1;   
        end
    end
        
    assign prediction = correct_predictions;
    assign completed = is_completed;

endmodule
