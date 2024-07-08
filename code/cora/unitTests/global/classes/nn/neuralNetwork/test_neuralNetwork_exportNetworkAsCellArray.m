function res = test_neuralNetwork_exportNetworkAsCellArray()
% test_neuralNetwork_exportNetworkAsCellArray - tests the 
%    exportNetworkAsCellArray function
%
% Syntax:
%    res = test_neuralNetwork_exportNetworkAsCellArray()
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
% Written:       02-May-2024
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

resvec = [];

% test reading basic network
filename = 'nn-nav-set.onnx';
nnOrg = neuralNetwork.readONNXNetwork(filename);
exportFileName = ['./' strrep(filename,'.onnx','.mat')];
exportNetworkAsCellArray(nnOrg, exportFileName);
load(exportFileName,'W','b','actFun');
nnNew = neuralNetwork.getFromCellArray(W,b,actFun);
xs = rand(nnOrg.neurons_in,10);
resvec(end+1) = compareMatrices(nnOrg.evaluate(xs),nnNew.evaluate(xs),1e-12);
delete(exportFileName);

% gather results
res = all(resvec);

end

% ------------------------------ END OF CODE ------------------------------
