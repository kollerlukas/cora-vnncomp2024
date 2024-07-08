function res = testnn_neuralNetwork_exportONNXNetwork()
% testnn_neuralNetwork_exportONNXNetwork - tests the readONNXNetwork function
%
% Syntax:
%    res = testnn_neuralNetwork_exportONNXNetwork()
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
% See also: -

% Authors:       Tobias Ladner
% Written:       30-April-2024
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

resvec = [];

% test reading basic network
filename = 'nn-nav-set.onnx';
nnOrg = neuralNetwork.readONNXNetwork(filename);
exportFileName = ['./' filename];
exportONNXNetwork(nnOrg, exportFileName);
nnNew = neuralNetwork.readONNXNetwork(filename);
xs = rand(nnOrg.neurons_in,10);
resvec(end+1) = compareMatrices(nnOrg.evaluate(xs),nnNew.evaluate(xs),1e-12);
delete(exportFileName);

% gather results
res = all(resvec);

end

% ------------------------------ END OF CODE ------------------------------
