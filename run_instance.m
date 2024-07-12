function res = run_instance(benchName,modelPath,vnnlibPath,resultsPath, ...
    timeout,verbose)
    fprintf('run_instance(%s,%s,%s,%s,%d,%d)...\n',benchName,modelPath, ...
        vnnlibPath,resultsPath,timeout,verbose);
    try
        fprintf('--- Loading MATLAB file...');
        % Create filename.
        instanceFilename = ...
            getInstanceFilename(benchName,modelPath,vnnlibPath);
        % Load stored network and specification.
        load(instanceFilename,'nn','options','permuteInputDims', ...
            'X0','specs');
        fprintf(' done\n');
        
        fprintf('--- Deleting MATLAB file...');
        % Delete file with stored networks and specification.
        delete(instanceFilename);
        fprintf(' done\n');

        fprintf('--- Running verification...');
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
        
            while true
                try
                    % Do verification.
                    [res,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,true);
                    break;
                catch e
                    if ismember(e.identifier, ...
                            {'parallel:gpu:array:pmaxsize', ...
                                'parallel:gpu:array:OOM', ...
                                'MATLAB:array:SizeLimitExceeded'}) ...
                            && options.nn.train.mini_batch_size > 1
                        options.nn.train.mini_batch_size = ...
                            floor(1/2*options.nn.train.mini_batch_size);
                        fprintf('--- OOM error: half batchSize %d...\n', ...
                            options.nn.train.mini_batch_size);
                    else
                        % Print the error message. 
                        fprintf(newline);
                        fprintf(e.message);
                        fprintf(newline);
                        % No result.
                        res = [];
                        break;
                    end
                end
            end
            fprintf(' done\n');

            fprintf('Writing results...\n');
            fprintf('--- opening results file ...');
            % Open results file.
            fid = fopen(resultsPath,'w');
            fprintf(' done\n');

            fprintf('--- writing file ...');
            % Write results.
            if strcmp(res,'VERIFIED')
                res = 'unsat';
                % Write content.
                fprintf(fid,['unsat' newline]);
                fclose(fid);
            elseif strcmp(res,'COUNTER EXAMPLE')
                res = 'sat';
                % Reorder input dimensions...
                if permuteInputDims
                  inSize = nn.layers{1}.inputSize([2 1 3]);
                  x_ = reshape(permute(reshape(x_,inSize),[2 1 3]),[],1);
                end
                % Write content.
                fprintf(fid,['sat' newline '(']);
                % Write input values.
                for j=1:size(x_,1)
                    fprintf(fid,['(X_%d %f)' newline],j-1,x_(j));
                end
                % Write output values.
                for j=1:size(y_,1)
                    fprintf(fid,['(Y_%d %f)' newline],j-1,y_(j));
                end
                fprintf(fid,')');
                fclose(fid);
                % We found a counterexample; we dont have to check the other
                % input sets.
                break;
            else
                res = 'unknown';
                % We cannot verify an input set; we dont have to check the other
                % input sets.
                fprintf(fid,['unknown' newline]);
                fclose(fid);
                break;
            end
        end
        fprintf(' done\n');

    catch e
        fprintf(e.message);
        % There is some issue with the parsing; e.g. acasxu prop_6.vnnlib
        res = 'unknown';
        fprintf(' done\n');

        % Open results file.
        fid = fopen(resultsPath,'w');
        fprintf(fid,['unknown' newline]);
        fclose(fid);
    end

    if verbose
        % Print result.
        fprintf('%s -- %s: %s\n',modelPath,vnnlibPath,res);
        time = toc;
        fprintf('--- Verification time: %.4f / %.4f [s]\n',time,timeout);
    end

end
