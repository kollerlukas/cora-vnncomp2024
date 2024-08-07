function r = evaluateTaylm(obj, input, options)
% evaluateTaylm - evaluates the activation layer on a taylor model
%
% Syntax:
%    r = evaluateTaylm(obj, input, options)
%
% Inputs:
%    input - taylor model
%    options - options for network evaluation (stored in options.nn)
%
% Outputs:
%    r - updated taylor model
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: nnLayer

% Authors:       Niklas Kochdumper, Tobias Ladner
% Written:       17-September-2021
% Last update:   ---
% Last revision: 05-April-2022 (TL)

% ------------------------------ BEGIN CODE -------------------------------

% init order
if ~all(size(input) == size(obj.order))
    obj.order = ones(size(input)) * max(obj.order);
end

% loop over all neurons in the current layer
r = input;
for j = 1:size(input, 1)
    r(j) = obj.evaluateTaylmNeuron(r(j), obj.order(j), options);
end

end

% ------------------------------ END OF CODE ------------------------------
