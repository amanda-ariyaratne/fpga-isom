`timescale 1ns / 1ps

module tsom
        #(
        parameter DIM = 100,
        parameter LOG2_DIM = 7,    // log2(DIM)
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
        parameter ITERATION_NB_STEP = 1, // total_iterations / nb_radius_step
        
        parameter INITIAL_UPDATE_PROB = 1000,
        parameter UPDATE_PROB_STEP = 100,
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
    
    reg training_en = 0;
    reg next_iteration_en=0;
    reg next_x_en=0;    
    reg dist_enable = 0;
    reg init_neigh_search_en=0;  
    reg nb_search_en=0;
    reg test_en = 0;
    reg classify_x_en = 0;
    reg classify_weights_en = 0;
    reg init_classification_en=0;
    reg classification_en = 0;
    reg class_label_en=0;
    reg write_en = 0;
    reg lfsr_en = 1;
    reg seed_en = 1;
    reg is_completed = 0;
    
    ///////////////////////////////////////////////////////*******************Read weight vectors***********/////////////////////////////////////
    ///////////////////////////////////////////////////////*******************looping variables***********/////////////////////////////////////
    reg signed [LOG2_ROWS:0] i = 0;
    reg signed [LOG2_COLS:0] j = 0;
    reg signed [LOG2_DIM:0] k = DIM-1;
    reg signed [LOG2_DIM:0] kw = DIM-1;
    reg signed [LOG2_DIM:0] k1 = DIM-1;
    reg signed [LOG2_DIM:0] k2 = DIM-1;
    
    reg signed [LOG2_TRAIN_ROWS:0] t1 = 0;
    reg signed [LOG2_TEST_ROWS:0] t2 = 0;
    
    ///////////////////////////////////////////////////////*******************other variables***********/////////////////////////////////////
    reg signed [DIGIT_DIM-1:0] weights [ROWS-1:0][COLS-1:0][DIM-1:0];
    reg signed [DIGIT_DIM-1:0] trainX [TRAIN_ROWS-1:0][DIM-1:0];
    reg signed [DIGIT_DIM-1:0] testX [TEST_ROWS-1:0][DIM-1:0];
    
    reg [LOG2_NUM_CLASSES-1:0] trainY [TRAIN_ROWS-1:0];
    reg [LOG2_NUM_CLASSES-1:0] testY [TEST_ROWS-1:0];
    
    integer weights_file;
    integer trains_file;
    integer test_file;
    
    reg [(DIM*DIGIT_DIM)-1:0] rand_v;
    reg [(DIM*DIGIT_DIM)+LOG2_NUM_CLASSES-1:0] temp_train_v;
    reg [(DIM*DIGIT_DIM)+LOG2_NUM_CLASSES-1:0] temp_test_v;
    
    integer eof_weight;
    integer eof_train;
    integer eof_test;
    
    ///////////////////////////////////////////////////////*******************Initialize weight vectors***********/////////////////////////////////////
    
    initial begin
        weights_file = $fopen("/home/mad/Documents/fpga-isom/tsom/weights.data","r");
        while (!$feof(weights_file)) begin
            eof_weight = $fscanf(weights_file, "%b\n",rand_v);
            
            for(kw=DIM-1;kw>=0;kw=kw-1) begin
                weights[i][j][kw] = rand_v[(DIGIT_DIM*kw)+1-:DIGIT_DIM];
            end
            
            j = j + 1;
            if (j == COLS) begin
                j = 0;
                i = i + 1;
            end
        end
        $fclose(weights_file);
    end
    
    ///////////////////////////////////////////////////////*******************Initialize train vectors***********/////////////////////////////////////
    
    initial begin
        trains_file = $fopen("/home/aari/Projects/Vivado/fpga_som/tsom/train.data","r");
        while (!$feof(trains_file)) begin        
            eof_train = $fscanf(trains_file, "%b\n",temp_train_v);
            
            for(k1=DIM-1;k1>=0;k1=k1-1) begin
                trainX[t1][k1] = temp_train_v[(DIGIT_DIM*k1)+1+LOG2_NUM_CLASSES -:DIGIT_DIM];
            end
            trainY[t1] = temp_train_v[LOG2_NUM_CLASSES-1:0];
            t1 = t1 + 1;
        end
        $fclose(trains_file);        
    end
    
    ///////////////////////////////////////////////////////*******************Initialize test vectors***********/////////////////////////////////////
    
    initial begin
        test_file = $fopen("/home/aari/Projects/Vivado/fpga_som/tsom/test.data","r");
        while (!$feof(test_file)) begin
            eof_test = $fscanf(test_file, "%b\n",temp_test_v);
            for(k2=DIM-1;k2>=0;k2=k2-1) begin
                testX[t2][k2] = temp_test_v[(DIGIT_DIM*k2)+LOG2_NUM_CLASSES+1 -:DIGIT_DIM];
            end
                
            testY[t2] = temp_test_v[LOG2_NUM_CLASSES-1:0];
            t2 = t2 + 1;
        end
        $fclose(test_file);  
        training_en = 1;
    end
    
    //////////////////////////////////////////////////////********************Initialize frequenct list*************//////////////////////////////
    
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
        $display("class frequnecy list initialized");
        is_completed = 1;
    end
    
    ///////////////////////////////////////////////////////****************Start LFSR**************/////////////////////////////////////
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
    
    reg [LOG2_TEST_ROWS:0] correct_predictions = 0;
    
    assign prediction = correct_predictions;
    assign completed = is_completed;
    
endmodule
