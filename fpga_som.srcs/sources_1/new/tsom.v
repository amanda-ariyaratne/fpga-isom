`timescale 1ns / 1ps


module tsom
    #(
        parameter DIM = 1000,
        parameter LOG2_DIM = 10,    // log2(DIM)
        parameter DIGIT_DIM = 2,
        parameter signed k_value = 1,
        
        parameter ROWS = 5,
        parameter LOG2_ROWS = 3,   // log2(ROWS)
        parameter COLS = 5,
        parameter LOG2_COLS = 3,     
        
        parameter TRAIN_ROWS = 75,
        parameter LOG2_TRAIN_ROWS = 7, // log2(TRAIN_ROWS)
        parameter TEST_ROWS = 150,
        parameter LOG2_TEST_ROWS = 8,  // log2(TEST_ROWS)
        
        parameter NUM_CLASSES = 3+1,
        parameter LOG2_NUM_CLASSES = 1+1, // log2(NUM_CLASSES)  
        
        parameter TOTAL_ITERATIONS=4,              
        parameter LOG2_TOT_ITERATIONS = 4,
        
        parameter INITIAL_NB_RADIUS = 3,
        parameter NB_RADIUS_STEP = 1,
        parameter LOG2_NB_RADIUS = 3,
        parameter ITERATION_NB_STEP = 3, // total_iterations / nb_radius_step
        
        parameter INITIAL_UPDATE_PROB = 1000,
        parameter UPDATE_PROB_STEP = 200,
        parameter LOG2_UPDATE_PROB = 10,
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
    
    ///////////////////////////////////////////////////////*******************Other variables***********/////////////////////////////////////
    
    reg signed [LOG2_TOT_ITERATIONS:0] iteration;
    
    reg [LOG2_ROWS:0] ii = 0;
    reg [LOG2_COLS:0] jj = 0;
    reg [LOG2_NUM_CLASSES:0] kk = 0;
    
    reg [LOG2_COLS:0] bmu [1:0];
    reg [LOG2_TRAIN_ROWS:0] class_frequency_list [ROWS-1:0][COLS-1:0][NUM_CLASSES-1:0];
    
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
        weights_file = $fopen("/home/mad/Documents/fpga-isom/tsom/weights.data","r");
        while (!$feof(weights_file))
        begin
            eof_weight = $fscanf(weights_file, "%b\n",rand_v);
            
            for(kw=DIM-1;kw>=0;kw=kw-1)
            begin
                weights[i][j][kw] = rand_v[(DIGIT_DIM*kw)+1-:DIGIT_DIM];
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
        trains_file = $fopen("/home/mad/Documents/fpga-isom/tsom/train.data","r");
        while (!$feof(trains_file))
            begin        
            eof_train = $fscanf(trains_file, "%b\n",temp_train_v);
            
            for(k1=DIM-1;k1>=0;k1=k1-1)
            begin
                trainX[t1][k1] = temp_train_v[(DIGIT_DIM*k1)+1+LOG2_NUM_CLASSES -:DIGIT_DIM];
            end
            trainY[t1] = temp_train_v[LOG2_NUM_CLASSES-1:0];
            t1 = t1 + 1;
        end
        $fclose(trains_file);
        training_en = 1;
    end

    ///////////////////////////////////////////////////////*******************Read test vectors***********/////////////////////////////////////
    initial begin
        test_file = $fopen("/home/mad/Documents/fpga-isom/tsom/test.data","r");
        while (!$feof(test_file))
        begin
            eof_test = $fscanf(test_file, "%b\n",temp_test_v);
            for(k2=DIM-1;k2>=0;k2=k2-1)
            begin
                testX[t2][k2] = temp_test_v[(DIGIT_DIM*k2)+LOG2_NUM_CLASSES+1 -:DIGIT_DIM];
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
    
    always @(posedge clk)
    begin
        if (next_iteration_en)
        begin
            t1 = -1; // reset trainset pointer
            if (iteration<(TOTAL_ITERATIONS-1)) begin
                iteration = iteration + 1;
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
    
    always @(posedge clk)
    begin
        if (next_x_en && !classification_en)
        begin                
            if (t1<TRAIN_ROWS-1)
            begin        
                t1 = t1 + 1;
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
    reg [LOG2_DIM:0] min_distance = DIM;   
    reg [LOG2_DIM:0] distances [ROWS-1:0][COLS-1:0];       
    reg [LOG2_COLS:0] minimum_distance_indices [(ROWS*COLS-1):0][1:0];
    reg [LOG2_DIM-1:0] min_distance_next_index = 0;
    
    reg [LOG2_DIM:0] hash_count;    
    reg [LOG2_DIM:0] min_hash_count;
    reg [LOG2_DIM:0] hash_counts [ROWS-1:0][COLS-1:0]; 
        
    reg [LOG2_ROWS:0] idx_i;
    reg [LOG2_COLS:0] idx_j;   
    
    reg [DIGIT_DIM-1:0] w;      
    reg [DIGIT_DIM-1:0] x;      
    
    always @(posedge clk)
    begin
        if (dist_enable)
        begin
            i = 0;
            j = 0;
            k = 0;
            min_distance_next_index = 0; // reset index
            min_distance = DIM;
            for (i=0;i<ROWS;i=i+1)
            begin
                for (j=0;j<COLS;j=j+1)
                begin
                    hamming_distance = 0;
                    hash_count = 0;
                    for (k=0;k<DIM;k=k+1)
                    begin
                        w = weights[i][j][k];
                        x = trainX[t1][k];
                        
                        if (w==0 && x==1)
                            hamming_distance=hamming_distance+1;
                        else if (w==1 && x==0)
                            hamming_distance=hamming_distance+1;
                            
                        // get zero count
                        if (w==2)
                            hash_count=hash_count+1;  
                            
                    end // k
                    
                    distances[i][j] = hamming_distance;
                    hash_counts[i][j] = hash_count;
                    
                    // get minimum distance index list
                    if (min_distance == hamming_distance) begin
                        minimum_distance_indices[min_distance_next_index][1] = i;
                        minimum_distance_indices[min_distance_next_index][0] = j;
                        min_distance_next_index = min_distance_next_index + 1;
                    end
                    
                    if (min_distance>hamming_distance) begin
                        min_distance = hamming_distance;
                        minimum_distance_indices[0][1] = i;
                        minimum_distance_indices[0][0] = j;                        
                        min_distance_next_index = 1;
                    end
                end //j                
            end // i
            
            // if there are more than one bmu
            if (min_distance_next_index > 1) begin
                iii = 0;
                min_hash_count = DIM-1;
                for(iii=0;iii<min_distance_next_index; iii=iii+1) begin
                    idx_i = minimum_distance_indices[iii][1];
                    idx_j = minimum_distance_indices[iii][0];
                    // $display("more than one bmu ", min_distance_next_index, " iii ", iii);
                    if (hash_counts[idx_i][idx_j] < min_hash_count) begin
                        min_hash_count = hash_counts[idx_i][idx_j];
                        bmu[1] = idx_i;
                        bmu[0] = idx_j;
                    end
                end
            end
            
            
            else begin // only one minimum distance node is there 
                bmu[1] = minimum_distance_indices[0][1];
                bmu[0] = minimum_distance_indices[0][0];
            end
            
            dist_enable = 0;
            
            if (!classification_en)
                init_neigh_search_en = 1; // find neighbours
            else
                next_x_en = 1; // classify node
        end        
    end
    
    //////////////////////************Start Neighbourhood search************//////////////////////////////////////////
    
    reg signed [LOG2_ROWS+1:0] bmu_i;
    reg signed [LOG2_COLS+1:0] bmu_j;
    reg signed [LOG2_ROWS+1:0] bmu_x;
    reg signed [LOG2_COLS+1:0] bmu_y;
    reg signed [LOG2_NB_RADIUS+1:0] man_dist; /////////// not sure
    reg signed [LOG2_NB_RADIUS+1:0] nb_radius = INITIAL_NB_RADIUS;
    reg signed [LOG2_UPDATE_PROB+1:0] update_prob = INITIAL_UPDATE_PROB;
    integer signed step_i;
     
    // update update probability
    always @(posedge clk)
    begin
        for (step_i=1; step_i<=STEP;step_i = step_i+1)
        begin
            if ((iteration<(ITERATION_STEP*step_i)) && (iteration>=(ITERATION_STEP*(step_i-1))))
            begin
                update_prob <= UPDATE_PROB_STEP*(STEP-step_i+1);
            end
        end
    end
    
    // update neighbourhood radius
    always @(posedge clk)
    begin
        for (step_i=1; step_i<=4;step_i = step_i+1)
        begin
            if ( (iteration<(ITERATION_NB_STEP*step_i)) && (iteration>= (ITERATION_NB_STEP*(step_i-1)) ) ) begin
                nb_radius <=  NB_RADIUS_STEP*(4-step_i);
            end
        end
    end
    
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

    always @(posedge clk)
    begin    
        if (nb_search_en) begin  
            man_dist = (bmu_x-bmu_i) >= 0 ? (bmu_x-bmu_i) : (bmu_i-bmu_x);
            man_dist = man_dist + ((bmu_y - bmu_j)>= 0 ? (bmu_y - bmu_j) : (bmu_j - bmu_y));              
            
            if (man_dist <= nb_radius) begin
                // update neighbourhood
                for (digit=0; digit<DIM; digit=digit+1) begin
                   if (random_number_arr[RAND_NUM_BIT_LEN*digit +: RAND_NUM_BIT_LEN] < update_prob) begin                        
                        seed_en = 0;
                        w = weights[bmu_i][bmu_j][digit];
                        x = trainX[t1][digit];
                        
                        if (w==1 && x==0)
                            weights[bmu_i][bmu_j][digit]=2;
                        else if (w==0 && x==1)
                            weights[bmu_i][bmu_j][digit]=2;                            
                        else if (w==2)
                            weights[bmu_i][bmu_j][digit]=x;
                    end
                end                
            end
                        
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
            $display("class_label_en");   
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
                    if (class_labels[i][j] == NUM_CLASSES-1) /////////// hardcoded default value
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
                test_en = 0;
                write_en = 1;            
            end
        end
    end
    
    reg [LOG2_TEST_ROWS:0] correct_predictions = 0; // should take log2 of test rows
    reg [LOG2_NUM_CLASSES:0] predictionY[TEST_ROWS-1:0];
    
    reg [LOG2_TEST_ROWS:0] tot_predictions = 0;
    
    always @(posedge clk)
    begin
        if (classify_x_en)
        begin
            i = 0;
            j = 0;
            k = 0;
            min_distance_next_index = 0; // reset index
            min_distance = DIM;
            for (i=0;i<ROWS;i=i+1)
            begin
                for (j=0;j<COLS;j=j+1)
                begin
                    hamming_distance = 0;
                    hash_count = 0;
                    for (k=0;k<DIM;k=k+1)
                    begin
                        w = weights[i][j][k];
                        x = testX[t2][k];
                        
                        if (w==0 && x==1)
                            hamming_distance=hamming_distance+1;
                        else if (w==1 && x==0)
                            hamming_distance=hamming_distance+1;
                            
                        // get zero count
                        if (w==2)
                            hash_count=hash_count+1;  
                            
                    end // k
                    
                    distances[i][j] = hamming_distance;
                    hash_counts[i][j] = hash_count;
                    
                    // get minimum distance index list
                    if (min_distance == hamming_distance) begin
                        minimum_distance_indices[min_distance_next_index][1] = i;
                        minimum_distance_indices[min_distance_next_index][0] = j;
                        min_distance_next_index = min_distance_next_index + 1;
                    end
                    
                    if (min_distance>hamming_distance) begin
                        min_distance = hamming_distance;
                        minimum_distance_indices[0][1] = i;
                        minimum_distance_indices[0][0] = j;                        
                        min_distance_next_index = 1;
                    end
                end //j                
            end // i
            
            // if there are more than one bmu
            if (min_distance_next_index > 1) begin
                iii = 0;
                min_hash_count = DIM-1;
                for(iii=0;iii<min_distance_next_index; iii=iii+1) begin
                    idx_i = minimum_distance_indices[iii][1];
                    idx_j = minimum_distance_indices[iii][0];
                    // $display("more than one bmu ", min_distance_next_index, " iii ", iii);
                    if (hash_counts[idx_i][idx_j] < min_hash_count) begin
                        min_hash_count = hash_counts[idx_i][idx_j];
                        bmu[1] = idx_i;
                        bmu[0] = idx_j;
                    end
                end
            end
            
            else begin // only one minimum distance node is there 
                bmu[1] = minimum_distance_indices[0][1];
                bmu[0] = minimum_distance_indices[0][0];
            end
            
            // check correctness
            if (class_labels[bmu[1]][bmu[0]] == testY[t2])
            begin
                correct_predictions = correct_predictions + 1;                
            end
            
            $display("pred ", class_labels[bmu[1]][bmu[0]], " actual ", testY[t2]);   
             
            predictionY[t2] = class_labels[bmu[1]][bmu[0]];     
            classify_x_en = 0;
            test_en = 1;
        end        
    end
    
    integer fd;    
    always @(posedge clk) begin
        if (write_en) begin
            fd = $fopen("/home/mad/Documents/fpga-isom/tsom/weight_out.data", "w");
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
