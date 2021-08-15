`timescale 1ns / 1ps

module test#(
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
        parameter LOG2_NUM_CLASSES = 1+1
    )
    (
        input wire clk,
        output wire [LOG2_TEST_ROWS:0] prediction,
        output wire completed
    );

    ///////////////////////////////////////////////////////*******************Declare enables***********/////////////////////////////////////
    ///////////////////////////////////////////////////////*******************Other variables***********/////////////////////////////////////
    
    
    reg [LOG2_ROWS:0] ii = 0;
    reg [LOG2_COLS:0] jj = 0;
    reg [LOG2_NUM_CLASSES:0] kk = 0;
    
    reg signed [DIGIT_DIM-1:0] weights [ROWS-1:0][COLS-1:0][DIM-1:0];
    reg signed [DIGIT_DIM-1:0] trainX [TRAIN_ROWS-1:0][DIM-1:0];    
    reg signed [DIGIT_DIM-1:0] testX [TEST_ROWS-1:0][DIM-1:0];
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
        weights_file = $fopen("/home/mad/Documents/fpga-isom/isom/weights.data","r");
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
        trains_file = $fopen("/home/mad/Documents/fpga-isom/isom/train.data","r");
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
    end

    ///////////////////////////////////////////////////////*******************Read test vectors***********/////////////////////////////////////
    initial begin
        test_file = $fopen("/home/mad/Documents/fpga-isom/isom/test.data","r");
        while (!$feof(test_file)) begin
            eof_test = $fscanf(test_file, "%b\n",temp_test_v);
            for(k2=DIM-1;k2>=0;k2=k2-1) begin
                testX[t2][k2] = temp_test_v[(DIGIT_DIM*k2)+LOG2_NUM_CLASSES+1 -:DIGIT_DIM];
            end
                
            testY[t2] = temp_test_v[LOG2_NUM_CLASSES-1:0];
            t2 = t2 + 1;
        end
        $fclose(test_file);  
    end
    
    reg [3:0] pred;
    
    always @(posedge clk) begin
        pred = 2'b11;
    end
    
    assign prediction = pred;
endmodule
