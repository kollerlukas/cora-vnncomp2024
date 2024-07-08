function ls = Inf(n)
% Inf - instantiates a fullspace level set
%
% Syntax:
%    ls = levelSet.Inf(n)
%
% Inputs:
%    n - dimension
%
% Outputs:
%    ls - fullspace level set
%
% Example: 
%    ls = levelSet.Inf(2);
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:       Mark Wetzlinger
% Written:       09-January-2024
% Last update:   15-January-2024 (TL, parse input)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% parse input
if nargin == 0
    n = 0;
end
inputArgsCheck({{n,'att','numeric',{'scalar','nonnegative'}}});

% init levelSet
vars = sym('x',[n,1]);
eq = 0*vars - 1;
ls = levelSet(eq,vars,{"<="});

% ------------------------------ END OF CODE ------------------------------
