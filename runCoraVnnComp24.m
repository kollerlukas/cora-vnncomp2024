% Obtain all instances.
filename = 'instances.csv';
instances = readtable(filename,'Delimiter',',');
% Rename columns
instances.Properties.VariableNames = {'model','vnnlib','timeout'};

benchmarkNames = {};
models = {};
vnnlibs = {};
prepTimes = {};
results = {};
verifTimes = {};

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
% Specify whether the spacial dimensions of the input have to be flipped.
permuteInputDims = false;

for i=1:size(instances,1)
    % Extract current instance.
    instance = instances(i,:);
    modelPath = instance.model{1};
    vnnlibPath = instance.vnnlib{1};
    timeout = instance.timeout;

    % Load neural network.

    try
        % acasxu --------------------------------------------------------------
        % benchName = 'acasxu';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BSSC');
        % % Set the batch size.
        % options.nn.train.mini_batch_size = 512;
        % if strcmp(vnnlibPath,'vnnlib/prop_7.vnnlib') || ...
        %         strcmp(vnnlibPath,'vnnlib/prop_8.vnnlib')
        %     % Skip this instance.
        %     throw(CORAerror('CORA:notSupported'));
        % end
        % --- Comment: batchSize 512 (on my laptop). 
        
        % c_gan ---------------------------------------------------------------
        % benchName = 'c_gan';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BC');
        % --- TODO: implement convTranspose
        
        % collins_rul_cnn -----------------------------------------------------
        % benchName = 'collins_rul_cnn';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BCSS');
        % % Set the batch size.
        % options.nn.train.mini_batch_size = 128;
        % options.nn.interval_center = true;
        % permuteInputDims = true;
        % --- Comment: batchSize 256 (on my laptop).
    
        % cora ----------------------------------------------------------------
        % benchName = 'cora';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BC');
        % % Set the batch size.
        % options.nn.train.mini_batch_size = 128;
        % --- Comment: batchSize 128 (on my laptop). 
    
        % dist_shift ----------------------------------------------------------
        % benchName = 'dist_shift';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BC');
        % % Set the batch size.
        % options.nn.train.mini_batch_size = 256;
        % --- Comment: batchSize 256 (on my laptop). 
    
        % dist_shift_vnn_comp -------------------------------------------------
        % benchName = 'dist_shift_vnn_comp';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BC');
        % % Set the batch size.
        % options.nn.train.mini_batch_size = 256;
        % --- Comment: batchSize 256 (on my laptop). 
    
        % LinearizeNN ---------------------------------------------------------
        % benchName = 'LinearizeNN';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BC');
        % --- TODO: weird networks (MatMul) and concat => No
    
        % metaroom ------------------------------------------------------------
        % benchName = 'metaroom';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BCSS');
        % % Bring input into the correct shape.
        % permuteInputDims = true;
        % % Set the batch size.
        % options.nn.train.mini_batch_size = 1;
        % --- Comment: batchSize 1 (on my laptop); crashes?!; weird specs
    
        % nn4sys --------------------------------------------------------------
        % if ~strcmp(modelPath,'onnx/lindex.onnx') && ...
        %         ~strcmp(modelPath,'onnx/lindex_deep.onnx')
        %     continue
        % end
        % benchName = 'nn4sys';
        % % Set the batch size.
        % options.nn.train.mini_batch_size = 128;
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BC');
        % --- Comment: batchSize 128 (on my laptop); weird architectures
    
        % safeNLP -------------------------------------------------------------
        % benchName = 'safeNLP';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BC');
        % --- Comment: missing instances
    
        % tllverifybench ------------------------------------------------------
        % benchName = 'tllverifybench';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BC');
        % --- Comment: out of memory error; adaptively set batch size?
    
        % vnncomp2024_cifar100_benchmark --------------------------------------
        % benchName = 'vnncomp2024_cifar100_benchmark';
        % modelPath = ['onnx/' modelPath];
        % vnnlibPath = ['generated_vnnlib/' vnnlibPath];
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BCSS');
        % % Bring input into the correct shape.
        % permuteInputDims = true;
        % % Set the batch size and use interval-center.
        % options.nn.train.mini_batch_size = 8;
        % options.nn.interval_center = true;
        % --- Comment: point-eval possible; memory issues with zonotope eval.
    
        % vnncomp2024_tinyimagenet_benchmark ----------------------------------
        % benchName = 'vnncomp2024_tinyimagenet_benchmark';
        % modelPath = ['onnx/' modelPath];
        % vnnlibPath = ['generated_vnnlib/' vnnlibPath];
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BCSS');
        % % Bring input into the correct shape.
        % permuteInputDims = true;
        % % Set the batch size and use interval-center.
        % options.nn.train.mini_batch_size = 8;
        % options.nn.interval_center = true;
        % --- Comment: point-eval possible; memory issues with zonotope eval.
    
        % VNNComp2023_NN4Sys --------------------------------------------------
        % benchName = 'VNNComp2023_NN4Sys';
        % nn = neuralNetwork.readONNXNetwork(modelPath,false,'BCC');
        % --- Comment: weird architecture

     
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
        
            if strcmp(res,'VERIFIED')
                res = 'unsat';
            elseif strcmp(res,'COUNTER EXAMPLE')
                res = 'sat';
                % Produce witness file.
                modelName = regexp(modelPath,'([^/]+)(?=\.onnx$)','match');
                vnnlibName = regexp(vnnlibPath,'([^/]+)(?=\.vnnlib$)','match');
                witFilename = ['cora-results/' modelName{1} '_' ...
                    vnnlibName{1} '.counterexample'];
                % Open file.
                fid = fopen(witFilename,'w');
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
                % We found a counterexample; we dont have to check the other
                % input sets.
                break;
            else
                res = 'unknown';
                % We cannot verify an input set; we dont have to check the other
                % input sets.
                break;
            end
        end

    catch
        % There is some issue with the parsing; e.g. acasxu prop_6.vnnlib
        res = 'unknown';
    end

    % Print result.
    fprintf('%s -- %s: %s\n',modelPath,vnnlibPath,res);
    time = toc;
    fprintf('--- Verification time: %.4f / %.4f [s]\n',time,timeout);
    % Store outputs.
    benchmarkNames = [benchmarkNames; benchName];
    models = [models; modelPath];
    vnnlibs = [vnnlibs; vnnlibPath];
    prepTimes = [prepTimes; 0];
    results = [results; res];
    verifTimes = [verifTimes; time];
end

% Generate results table.
resultsTable = table(benchmarkNames,models,vnnlibs,prepTimes,results, ...
    verifTimes);
% Write to file.
writetable(resultsTable,'cora-results/results.csv');
