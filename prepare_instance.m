function res = prepare_instance(benchName,modelPath,vnnlibPath)

  try
      % Load neural network.
      [nn,options,permuteInputDims] = aux_readNetworkAndOptions( ...
          benchName,modelPath,vnnlibPath,false);
   
      % Load specification.
      [X0,specs] = vnnlib2cora(vnnlibPath);

      % Create filename.
      instanceFilename = getInstanceFilename(benchName,modelPath,vnnlibPath);

      % Store network, options, and specification.
      save(instanceFilename,'nn','options','permuteInputDims','X0','specs');
  catch
      % Some error
      res = 1;
      return;
  end
  res = 0;
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
