function res = run_instance(benchName,modelPath,vnnlibPath,resultsPath, ...
    timeout,verbose)

    try
        % Load neural network.
        [nn,options,permuteInputDims] = aux_readNetworkAndOptions( ...
            benchName,modelPath,vnnlibPath,false);
     
        % Load specification.
        [X0,specs] = vnnlib2cora(vnnlibPath);

        % Verify each input set individually.
        for j=1:length(X0)
    
            % Extract input set.
            x = 1/2*(X0{j}.sup + X0{j}.inf);
            r = 1/2*(X0{j}.sup - X0{j}.inf);
        
            if permuteInputDims
                inSize = nn.layers{1}.inputSize([2 1 3]);
                x = reshape(permute(reshape(x,inSize),[2 1 3]),[],1);
                r = reshape(permute(reshape(r,inSize),[2 1 3]),[],1);
            end
    
            % Extract specification.
            if isa(specs.set,'halfspace')
                A = specs.set.c';
                b = -specs.set.d;
            else
                A = specs.set.A;
                b = -specs.set.b;
            end
            safeSet = strcmp(specs.type,'safeSet');
        
            % Measure verification time.
            tic
        
            % Do verification.
            [res,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,true);

            % Open results file.
            fid = fopen(resultsPath,'w');

            % Write results.
            if strcmp(res,'VERIFIED')
                res = 'unsat';
                % Write content.
                fprintf(fid,'unsat\n');
                fclose(fid)
            elseif strcmp(res,'COUNTER EXAMPLE')
                res = 'sat';
                % TODO: reorder input dimensions...
                % Write content.
                fprintf(fid,'sat\n(');
                % Write input values.
                for j=1:size(x_,1)
                    fprintf(fid,'(X_%d %f)\n',j-1,x_(j));
                end
                % Write output values.
                for j=1:size(y_,1)
                    fprintf(fid,'(Y_%d %f)\n',j-1,y_(j));
                end
                fprintf(fid,')');
                fclose(fid)
                % We found a counterexample; we dont have to check the other
                % input sets.
                break;
            else
                res = 'unknown';
                % We cannot verify an input set; we dont have to check the other
                % input sets.
                fprintf(fid,'unknown\n');
                fclose(fid)
                break;
            end
        end

    catch
        % There is some issue with the parsing; e.g. acasxu prop_6.vnnlib
        res = 'unknown';
    end

    if verbose
        % Print result.
        fprintf('%s -- %s: %s\n',modelPath,vnnlibPath,res);
        time = toc;
        fprintf('--- Verification time: %.4f / %.4f [s]\n',time,timeout);

        disp(fileread(resultsPath))
    end

end

% Auxiliary functions -----------------------------------------------------

function [nn,options,permuteInputDims] = aux_readNetworkAndOptions( ...
    benchName,modelPath,vnnlibPath,verbose)

    % Create evaluation options.
    options.nn = struct(...
        'use_approx_error',true,...
        'poly_method','bounds',...'bounds','singh'
        'train',struct(...
            'backprop',false,...
            'mini_batch_size',128 ...
        ) ...
    );
    % Set default training parameters
    options = nnHelper.validateNNoptions(options,true);
    options.nn.interval_center = false;

    if strcmp(benchName,'test')
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'','', ...
            'dlnetwork',false);
        % Set the batch size.
        options.nn.train.mini_batch_size = 512;
        permuteInputDims = false;
    elseif strcmp(benchName,'acasxu_2023')
        % acasxu ----------------------------------------------------------
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BSSC');
        % Set the batch size.
        options.nn.train.mini_batch_size = 512;
        permuteInputDims = false;
        if strcmp(vnnlibPath,'vnnlib/prop_7.vnnlib') || ...
                strcmp(vnnlibPath,'vnnlib/prop_8.vnnlib')
            % Skip this instance.
            throw(CORAerror('CORA:notSupported',...
                sprintf("Instance '%s' of benchmark '%s' is not " + ...
                "supported!",vnnlibPath,benchName)));
        end
        % --- Comment: batchSize 512 (on my laptop). 
    elseif strcmp(benchName,'cctsdb_yolo_2023')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'cgan_2023')
        % c_gan -----------------------------------------------------------
        % benchName = 'c_gan';
        % nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
        % --- TODO: implement convTranspose
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'cifar100')
        % vnncomp2024_cifar100_benchmark ----------------------------------
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
        % Bring input into the correct shape.
        permuteInputDims = true;
        % Set the batch size and use interval-center.
        options.nn.train.mini_batch_size = 8;
        options.nn.interval_center = true;
        % --- Comment: point-eval possible; memory issues with zonotope eval.
    elseif strcmp(benchName,'collins_aerospace_benchmark')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'collins_rul_cnn_2023')
        % collins_rul_cnn -------------------------------------------------
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
        % Set the batch size.
        options.nn.train.mini_batch_size = 128;
        options.nn.interval_center = true;
        permuteInputDims = true;
        % --- Comment: batchSize 128 (on my laptop).
    elseif strcmp(benchName,'collins_yolo_robustness_2023')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'dist_shift_2023')
        % dist_shift ------------------------------------------------------
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
        % Set the batch size.
        options.nn.train.mini_batch_size = 256;
        permuteInputDims = false;
        % --- Comment: batchSize 256 (on my laptop). 
    elseif strcmp(benchName,'linearizenn')
        % LinearizeNN -----------------------------------------------------
        % benchName = 'LinearizeNN';
        % nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
        % --- TODO: weird networks (MatMul) and concat => No
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'lsnc')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'metaroom_2023')
        % metaroom --------------------------------------------------------
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
        % Set the batch size.
        options.nn.train.mini_batch_size = 1;
        % Bring input into the correct shape.
        permuteInputDims = true;
        % --- Comment: 
    elseif strcmp(benchName,'ml4acopf_2023')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'ml4acopf_2024')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'nn4sys_2023')
        % nn4sys ----------------------------------------------------------
        if ~strcmp(modelPath,'onnx/lindex.onnx') && ...
                ~strcmp(modelPath,'onnx/lindex_deep.onnx')
            % Skip this instance.
            throw(CORAerror('CORA:notSupported',...
                sprintf("Model '%s' of benchmark '%s' is not " + ...
                "supported!",modelPath,benchName)));
        end
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
        % Set the batch size.
        options.nn.train.mini_batch_size = 128;
        permuteInputDims = false;
        % --- Comment: batchSize 128 (on my laptop); weird architectures
    elseif strcmp(benchName,'safenlp')
        % safeNLP ---------------------------------------------------------
        % benchName = 'safeNLP';
        % nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
        % --- Comment: missing instances
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'tinyimagenet')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'tllverifybench_2023')
        % tllverifybench --------------------------------------------------
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
        % --- Comment: out of memory error; adaptively set batch size?
    elseif strcmp(benchName,'traffic_signs_recognition_2023')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'vggnet16_2023')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'vit_2023')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    elseif strcmp(benchName,'yolo_2023')
        throw(CORAerror('CORA:notSupported',...
            sprintf("Benchmark '%s' not supported!",benchName)));
    else
        throw(CORAerror('CORA:notSupported',...
            sprintf("Unknown benchmark '%s'!",benchName)));
    end
    
        % cora ------------------------------------------------------------
        % benchName = 'cora';
        % nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
        % % Set the batch size.
        % options.nn.train.mini_batch_size = 128;
        % --- Comment: batchSize 128 (on my laptop).
    
        % vnncomp2024_tinyimagenet_benchmark ------------------------------
        % benchName = 'vnncomp2024_tinyimagenet_benchmark';
        % modelPath = ['onnx/' modelPath];
        % vnnlibPath = ['generated_vnnlib/' vnnlibPath];
        % nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
        % % Bring input into the correct shape.
        % permuteInputDims = true;
        % % Set the batch size and use interval-center.
        % options.nn.train.mini_batch_size = 8;
        % options.nn.interval_center = true;
        % --- Comment: point-eval possible; memory issues with zonotope eval.

end
