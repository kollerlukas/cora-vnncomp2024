function res = test_polyZonotope_relaxExponents
% test_polyZonotope_relaxExponents - unit test function for exponent relaxation
%
% Syntax:
%    res = test_polyZonotope_relaxExponents
%
% Inputs:
%    -
%
% Outputs:
%    res - true/false 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% Authors:       Tobias Ladner
% Written:       25-January-2024
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

resvec = [];

% test simple cases
pZ = polyZonotope([1;1]);
pZ_relax = pZ.relaxExponents(1);
resvec(end+1) = isequal(pZ,pZ_relax);

pZ = polyZonotope(zonotope([1;1],[2 4; 5 2]));
pZ_relax = pZ.relaxExponents(1);
resvec(end+1) = isequal(pZ,pZ_relax);

% test paper
pZ = polyZonotope([2;2],[1 -1; 2 3],[],[1 1; 4 2]);
pZ_relax = pZ.relaxExponents(1);
pZ_exp = polyZonotope([2;2],[0;5],[0.25;0.5],[1;2]);
resvec(end+1) = isequal(pZ_relax,pZ_exp);

% test graphs
pZ = polyZonotope([2;2],[1 -1; 2 3],[],[1 1; 4 2]);
[~,G] = pZ.relaxExponents(1,'greedy');
resvec(end+1) = isa(G,'digraph');

pZ = polyZonotope([2;2],[1 -1; 2 3],[],[1 1; 4 2]);
[~,G] = pZ.relaxExponents(1,'all');
resvec(end+1) = isa(G,'graph');

% gather results
res = all(resvec);

% ------------------------------ END OF CODE ------------------------------
