`timescale 1ns / 1ps


module som
    #(
        parameter DIM = 10,
        parameter LOG2_DIM = 4,    // log2(DIM)
        parameter DIGIT_DIM = 2,
        parameter signed k_value = 1,
        parameter ROWS = 5,
        parameter LOG2_ROWS = 3,   // log2(ROWS)
        parameter COLS = 5,
        parameter LOG2_COLS = 3,   // log2(COLS)
        parameter TRAIN_ROWS = 75,
        parameter LOG2_TRAIN_ROWS = 7, // log2(TRAIN_ROWS)
        parameter TEST_ROWS = 150,
        parameter LOG2_TEST_ROWS = 8,  // log2(TEST_ROWS)
        parameter NUM_CLASSES = 3+1,
        parameter LOG2_NUM_CLASSES = 1+1, // log2(NUM_CLASSES)  
        parameter INITIAL_NB_RADIUS = 2,
        parameter LOG2_NB_RADIUS = 1
    )
    (
        input wire clk,
        output wire [LOG2_TEST_ROWS:0] prediction
    );

    ///////////////////////////////////////////////////////*******************Declare enables***********/////////////////////////////////////

    reg dist_enable = 0;
    reg init_neigh_search_en=0;  
    reg nb_search_en=0;
    reg next_x_en = 0;
    reg test_en = 0;
    reg classify_x_en = 0;
    reg classify_weights_en = 0;
    reg init_classification_en=0;
    reg class_label_en=0;
    
    ///////////////////////////////////////////////////////*******************Read weight vectors***********/////////////////////////////////////
    
    
    reg signed [DIGIT_DIM-1:0] weights [ROWS-1:0][COLS-1:0][DIM-1:0];
    reg [LOG2_ROWS:0] i = 0;
    reg [LOG2_COLS:0] j = 0;
    reg signed [LOG2_DIM:0] k = DIM-1;
    reg signed [LOG2_DIM:0] kw = DIM-1;
    reg signed [LOG2_DIM:0] k1 = DIM-1;
    reg signed [LOG2_DIM:0] k2 = DIM-1;
    
    
    integer weights_file;
    integer trains_file;
    integer test_file;
    reg [(DIM*2)-1:0] rand_v;
    integer eof_weight;
    
    initial
    begin
        weights_file = $fopen("/home/aari/Projects/Vivado/fpga_som/weights.data","r");
        eof_weight = 0;
        while (!eof_weight)
        begin
            eof_weight = $fscanf(weights_file, "%b\n",rand_v);
            
            for(kw=DIM-1;kw>=0;kw=kw-1)
            begin
                $display("wwww", kw);
                weights[i][j][kw] = rand_v[(2*kw)+1-:2];
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

    reg signed [DIGIT_DIM-1:0] trainX [TRAIN_ROWS-1:0][DIM-1:0];
    reg [LOG2_NUM_CLASSES-1:0] trainY [TRAIN_ROWS-1:0];
    reg signed [LOG2_TRAIN_ROWS:0] t1 = 0;
    reg [(DIM*DIGIT_DIM)+LOG2_NUM_CLASSES-1:0] temp_train_v;
    integer eof_train;
    
    initial
    begin
        trains_file = $fopen("/home/aari/Projects/Vivado/fpga_som/train.data","r");
        eof_train=0;
        while (eof_train!=1)   
            begin        
            eof_train = $fscanf(trains_file, "%b\n",temp_train_v);
            
            for(k1=DIM-1;k1>=0;k1=k1-1)
            begin
                $display("k1 ", k1);                
                trainX[t1][k1] = temp_train_v[(DIGIT_DIM*k1)+1+LOG2_NUM_CLASSES-:DIGIT_DIM];
            end
            trainY[t1] = temp_train_v[LOG2_NUM_CLASSES-1:0];
            t1 = t1 + 1;
        end
        $fclose(trains_file);
    end

    ///////////////////////////////////////////////////////*******************Read test vectors***********/////////////////////////////////////
    
    reg signed [DIGIT_DIM-1:0] testX [TEST_ROWS-1:0][DIM-1:0];
    reg [LOG2_NUM_CLASSES-1:0] testY [TEST_ROWS-1:0];
    reg signed [LOG2_TEST_ROWS:0] t2 = 0;
    reg [(DIM*DIGIT_DIM)+LOG2_NUM_CLASSES-1:0] temp_test_v;
    integer eof_test;
    
    initial
    begin
        test_file = $fopen("/home/aari/Projects/Vivado/fpga_som/test.data","r");
        eof_test = 0;
        while (!eof_test)
        begin
            eof_test = $fscanf(test_file, "%b\n",temp_test_v);
            for(k2=DIM-1;k2>=0;k2=k2-1)
            begin
                $display("k2 ", k2);
                testX[t2][k2] = temp_test_v[(DIGIT_DIM*k2)+1+LOG2_NUM_CLASSES-:DIGIT_DIM];
            end
                
                
            testY[t2] = temp_test_v[LOG2_NUM_CLASSES-1:0];
            t2 = t2 + 1;
        end
        $fclose(test_file);
        t1 = -1;
        next_x_en = 1;
    end
    
    ////////////////////*****************************Initialize frequenct list*************//////////////////////////////
    
    reg [LOG2_TRAIN_ROWS:0] class_frequency_list [ROWS-1:0][COLS-1:0][NUM_CLASSES-1:0];
    reg [LOG2_ROWS:0] ii = 0;
    reg [LOG2_COLS:0] jj = 0;
    reg [LOG2_NUM_CLASSES:0] kk = 0;
    
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
    end

    ///////////////////////////////////////////////////////*******************Start Training***********/////////////////////////////////////
    
    
    reg [LOG2_DIM:0] hamming_distance;
    reg [LOG2_DIM:0] non_zero_count;
    reg [LOG2_DIM:0] min_distance = DIM;    
    reg [LOG2_DIM:0] max_l0_norm;
    reg [LOG2_DIM:0] distances [ROWS-1:0][COLS-1:0]; // log2 dim
    reg [LOG2_DIM:0] l0_norms [ROWS-1:0][COLS-1:0];    
    reg [LOG2_COLS:0] minimum_distance_indices [(ROWS*COLS-1):0][1:0]; // since COL==ROW any of LOG2_can be used here
    reg [LOG2_ROWS:0] idx_i;
    reg [LOG2_COLS:0] idx_j;
    reg [LOG2_DIM-1:0] min_distance_next_index = 0;    
    reg [LOG2_COLS:0] bmu [1:0]; // since COL==ROW any of LOG2_ can be used here
    
    always @(posedge clk)
    begin
        if (next_x_en)
        begin
            if (init_classification_en)
                classify_weights_en = 1;
                
            if (t1<TRAIN_ROWS-1)
            begin              
                $display("Started...");  
                t1 = t1 + 1;
                next_x_en = 0;
                dist_enable = 1;
            end
            
            else
            begin
                next_x_en = 0;     
                t1 = 0;
                // off classification
                if (init_classification_en)
                begin
                    init_classification_en = 0;
                    dist_enable = 0;
                    class_label_en = 1;
                end
                //on classification
                else
                begin                    
                    init_classification_en = 1;
                    dist_enable = 1;
                end    
            end
        end
    end
    
    //////////////////******************************Find BMU******************************/////////////////////////////////
    
    always @(posedge clk)
    begin
        if (dist_enable)
        begin
            i = 0;
            j = 0;
            k = 0;
            for (i=0;i<ROWS;i=i+1)
            begin
                for (j=0;j<COLS;j=j+1)
                begin
                    hamming_distance = 0;
                    non_zero_count = 0;
                    for (k=0;k<DIM;k=k+1)
                    begin
                        // get distnace
                        hamming_distance = hamming_distance + (weights[i][j][k]*trainX[t1][k] == 2'b11 ? 2'b01 : 2'b00);
                        // get zero count
                        non_zero_count = non_zero_count + (weights[i][j][k] == 2'b00 ? 2'b00 : 2'b01); 
                    end // k
                    
                    distances[i][j] = hamming_distance;
                    l0_norms[i][j] = non_zero_count;
                    
                    // get minimum distance index list
                    if (min_distance == hamming_distance)
                    begin
                        minimum_distance_indices[min_distance_next_index][1] = i;
                        minimum_distance_indices[min_distance_next_index][0] = j;
                        min_distance_next_index = min_distance_next_index + 1;
                    end
                    
                    if (min_distance>hamming_distance)
                    begin
                        min_distance = hamming_distance;
                        minimum_distance_indices[0][1] = i;
                        minimum_distance_indices[0][0] = j;                        
                        min_distance_next_index = 1;
                    end
                end //j                
            end // i
            
            // if there are more than one bmu
            if (min_distance_next_index > 1)
            begin
                i = 0;
                max_l0_norm = 0;
                for(i=0;i<min_distance_next_index; i=i+1)
                begin
                    idx_i = minimum_distance_indices[i][1];
                    idx_j = minimum_distance_indices[i][0];
                    
                    if (l0_norms[idx_i][idx_j] > max_l0_norm)
                    begin
                        max_l0_norm = l0_norms[idx_i][idx_j];
                        bmu[1] = idx_i;
                        bmu[0] = idx_j;
                    end
                end
            end
            
            else // only one minimum distance node is there
            begin
                bmu[1] = minimum_distance_indices[0][1];
                bmu[0] = minimum_distance_indices[0][0];
            end
            dist_enable = 0;
            
            
            if (!init_classification_en)
                init_neigh_search_en = 1;
            else
                next_x_en = 1;
        end        
    end
    
    //////////////////////************Start Neighbourhood search************//////////////////////////////////////////
    
    reg signed [LOG2_ROWS+1:0] bmu_i;
    reg signed [LOG2_COLS+1:0] bmu_j;
    reg signed [LOG2_ROWS+1:0] bmu_x;
    reg signed [LOG2_COLS+1:0] bmu_y;
    reg signed [LOG2_NB_RADIUS+1:0] man_dist; /////////// not sure
    reg signed [LOG2_NB_RADIUS+1:0] nb_radius;
    integer digit;
    
    always @(posedge clk)
    begin    
        if (init_neigh_search_en)
        begin
            bmu_x = bmu[1]; bmu_y = bmu[0];    
            bmu_i = (bmu_x-LOG2_NB_RADIUS) < 0 ? 0 : (bmu_x-LOG2_NB_RADIUS);
            bmu_j = (bmu_y-LOG2_NB_RADIUS) < 0 ? 0 : (bmu_y-LOG2_NB_RADIUS);
            nb_radius = INITIAL_NB_RADIUS;
            init_neigh_search_en=0;
            nb_search_en=1;
        end
    end
    
    reg signed [2*DIM:0] temp[DIM-1:0];
    
    always @(posedge clk)
    begin    
        if (nb_search_en)
        begin   
            man_dist = (bmu_x-bmu_i) > 0 ? (bmu_x-bmu_i) : (bmu_i-bmu_x);
            man_dist = man_dist + ((bmu_y - bmu_j)>0 ? (bmu_y - bmu_j) : (bmu_j - bmu_y));  
            if (man_dist <= nb_radius)
            begin
                // update neighbourhood
                for (digit=0; digit<DIM; digit=digit+1)
                begin
                    temp[digit] = weights[bmu_i][bmu_j][digit] + trainX[t1][digit];
                    $display("wei ", weights[bmu_i][bmu_j][digit], " x ", trainX[t1][digit]);
                    
                    if (temp[digit]>k_value) 
                        weights[bmu_i][bmu_j][digit] = k_value;
                    else if (temp[digit]< -k_value) 
                        weights[bmu_i][bmu_j][digit] = -k_value;
                    else 
                        weights[bmu_i][bmu_j][digit] = temp[digit];
                    $display("wei ", weights[bmu_i][bmu_j][digit]);
                end
            end
            
            bmu_j = bmu_j + 1;            
            if (bmu_j == bmu_y+nb_radius+1 || bmu_j == COLS)
            begin
                bmu_j = (bmu_y-nb_radius) < 0 ? 0 : (bmu_y-nb_radius);
                bmu_i = bmu_i + 1;
            end            
            if (bmu_i == bmu_x+nb_radius+1 || bmu_i==ROWS)
            begin                
                nb_search_en = 0; // neighbourhood search finished        
                next_x_en = 1; // go to the next input
            end
        end
    end
    
    /////////////////////************Start Classification of weight vectors********///////////////////////
    
    reg [LOG2_NUM_CLASSES:0] class_labels [ROWS-1:0][COLS-1:0];    

    always @(posedge clk)
    begin
        if (classify_weights_en)
        begin
            class_frequency_list[bmu[1]][bmu[0]][trainY[t1]] =  class_frequency_list[bmu[1]][bmu[0]][trainY[t1]] + 1;
            classify_weights_en = 0;
        end
    end
    
    integer most_freq = 0;
    
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
                    class_labels[i][j] = 2; /////////// hardcoded 2
                    for(k=0;k<NUM_CLASSES;k=k+1)
                    begin
                        if (class_frequency_list[i][j][k]>most_freq)
                        begin
                            class_labels[i][j] = k;
                            most_freq = class_frequency_list[i][j][k];
                        end
                            
                    end
                end
            end
            class_label_en = 0;
            test_en = 1;
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
                test_en = 0;
                classify_x_en = 1;
            end            
            else
                test_en = 0; 
        end
    end
    
    reg [LOG2_TEST_ROWS-1:0] correct_predictions = 0; // should take log2 of test rows
    
    always @(posedge clk)
    begin
        if (classify_x_en)
        begin
            i = 0;
            j = 0;
            k = 0;
            min_distance_next_index = 0;  
            for (i=0;i<ROWS;i=i+1)
            begin
                for (j=0;j<COLS;j=j+1)
                begin
                    hamming_distance = 0;
                    non_zero_count = 0;
                    for (k=0;k<DIM;k=k+1)
                    begin
                        // get distnace
                        hamming_distance = hamming_distance + (weights[i][j][k]*testX[t2][k] == 2'b11 ? 2'b01 : 2'b00);
                        // get zero count
                        non_zero_count = non_zero_count + (weights[i][j][k] == 2'b00 ? 2'b00 : 2'b01); 
                    end // k
                    
                    distances[i][j] = hamming_distance;
                    l0_norms[i][j] = non_zero_count;
                    
                    // get minimum distance index list
                    if (min_distance == hamming_distance)
                    begin
                        minimum_distance_indices[min_distance_next_index][1] = i;
                        minimum_distance_indices[min_distance_next_index][0] = j;
                        min_distance_next_index = min_distance_next_index + 1;
                    end
                    
                    if (min_distance>hamming_distance)
                    begin
                        min_distance = hamming_distance;
                        minimum_distance_indices[0][1] = i;
                        minimum_distance_indices[0][0] = j;                        
                        min_distance_next_index = 1;
                    end
                end //j                
            end // i
            
            // if there are more than one bmu
            if (min_distance_next_index > 1)
            begin
                i = 0;
                max_l0_norm = 0;
                for(i=0;i<min_distance_next_index; i=i+1)
                begin
                    idx_i = minimum_distance_indices[i][1];
                    idx_j = minimum_distance_indices[i][0];
                    
                    if (l0_norms[idx_i][idx_j] > max_l0_norm)
                    begin
                        max_l0_norm = l0_norms[idx_i][idx_j];
                        bmu[1] = idx_i;
                        bmu[0] = idx_j;
                    end
                end
            end
            
            else // only one minimum distance node is there
            begin
                bmu[1] = minimum_distance_indices[0][1];
                bmu[0] = minimum_distance_indices[0][0];
            end
            // check correctness
            if (class_labels[bmu[1]][bmu[0]] == testY[t2])
                correct_predictions = correct_predictions + 1;
            
            classify_x_en = 0;
            test_en = 1;
        end        
    end
        
    assign prediction = correct_predictions;

endmodule
