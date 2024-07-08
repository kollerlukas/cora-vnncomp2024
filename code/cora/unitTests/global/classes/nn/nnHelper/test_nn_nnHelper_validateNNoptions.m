function res = test_nn_nnHelper_validateNNoptions()
% test_nn_nnHelper_validateNNoptions - tests the params validation for
%    the set-based neural network validation.
%
% Syntax:
%    res = test_nn_nnHelper_validateNNoptions()
%
% Inputs:
%    -
%
% Outputs:
%    res - boolean 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: nnHelper/validateNNoptions

% Authors:       Tobias Ladner
% Written:       29-November-2022
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

res = true;

% enter empty
options = struct;
options = nnHelper.validateNNoptions(options);

% enter full
options = nnHelper.validateNNoptions(options);

% test fields
options = struct; % test default
options = nnHelper.validateNNoptions(options); 
res = res && strcmp(options.nn.poly_method, 'regression');

options.nn.bound_approx = false;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.bound_approx == false;

options.nn.num_generators = 1000;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.num_generators == 1000;

options.nn.max_gens_post = 100;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.max_gens_post == 100;

options.nn.add_approx_error_to_GI = true;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.add_approx_error_to_GI == true;

options.nn.plot_multi_layer_approx_info = true;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.plot_multi_layer_approx_info == true;

options.nn.poly_method = 'ridgeregression';
options = nnHelper.validateNNoptions(options);
res = res && strcmp(options.nn.poly_method, 'ridgeregression');

options.nn.reuse_bounds = true;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.reuse_bounds == true;

options.nn.max_bounds = 2;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.max_bounds == 2;

options.nn.do_pre_order_reduction = false;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.do_pre_order_reduction == false;

options.nn.remove_GI = false;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.remove_GI == false;

options.nn.force_approx_lin_at = 2;
options = nnHelper.validateNNoptions(options);
res = res && options.nn.force_approx_lin_at == 2;


% test error message
try
    options = struct;
    options.nn.poly_method = 'unknown';
    options = nnHelper.validateNNoptions(options);
    % should have trown an error
    res = false;
end

end

% ------------------------------ END OF CODE ------------------------------
