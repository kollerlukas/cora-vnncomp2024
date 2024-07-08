function version = CORAVERSION()
% CORAVERSION - returns the current CORA version
%
% Syntax:
%    version = CORAVERSION()
%
% Inputs:
%    -
%
% Outputs:
%    version (string) - version of CORA
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: ---

% Authors:       Tobias Ladner
% Written:       07-August-2023
% Last update:   27-May-2024
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

version = 'CORA v2024.2.1';

% add suffix for development
version = [version ' (dev)'];

% ------------------------------ END OF CODE ------------------------------