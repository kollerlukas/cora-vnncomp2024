classdef nnTanhLayer < nnSShapeLayer
% nnTanhLayer - class for tanh layers
%
% Syntax:
%    obj = nnTanhLayer(name)
%
% Inputs:
%    name - name of the layer, defaults to type
%
% Outputs:
%    obj - generated object
%
% References:
%    [1] Koller, L. "Co-Design for Training and Verifying Neural Networks",
%           Master's Thesis
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork

% Authors:       Tobias Ladner, Lukas Koller
% Written:       28-March-2022
% Last update:   25-May-2023 (LK, approx error: closed form for order=1)
% Last revision: 10-August-2022 (renamed)

% ------------------------------ BEGIN CODE -------------------------------

properties
    % found via obj.findRegionPolys
    reg_polys = [ ...
     struct( ...
        'l',             -Inf, ...
        'u',               -5, ...
        'p', [                0, -0.9999546021312975 ], ...
        'd', 4.539786870244589e-05 ...
    ),struct( ...
        'l',               -5, ...
        'u',             -2.5, ...
        'p', [ 0.0003632483926581774, 0.007774061031399077, 0.06694242814327564, 0.2908011544996281, 0.6397668761140367, -0.4269522964773172 ], ...
        'd', 4.001313871632407e-05 ...
    ),struct( ...
        'l',             -2.5, ...
        'u',           -0.625, ...
        'p', [ -0.006746596773312201, -0.06833149578555836, -0.2607942809217674, -0.3930263398561141, 0.1220772321097194, 1.101216006513477, 0.02375333465213731 ], ...
        'd', 7.14657357248825e-05 ...
    ),struct( ...
        'l',           -0.625, ...
        'u',             1.25, ...
        'p', [ -0.005594453255578369, -0.02725315180792373, 0.1104113732706064, 0.01648843027614146, -0.329908142877703, -0.00249908438227904, 0.9998499350506984, 7.433106151624473e-05 ], ...
        'd', 9.677882028347937e-05 ...
    ),struct( ...
        'l',             1.25, ...
        'u',                5, ...
        'p', [ 5.120357844960832e-05, -0.001455636968522901, 0.01784420985752258, -0.1227508298674645, 0.514304570165882, -1.320606638452662, 1.938513354008329, -0.2654008527398747 ], ...
        'd', 7.128690805658287e-05 ...
    ),struct( ...
        'l',                5, ...
        'u',              Inf, ...
        'p', [                0, 0.9999546021312975 ], ...
        'd', 4.539786870244589e-05 ...
    ); ...
    ]
end

methods
    % constructor
    function obj = nnTanhLayer(name)
        if nargin < 1
            name = [];
        end
        % call super class constructor
        obj@nnSShapeLayer(name)
    end
end

% evaluate ----------------------------------------------------------------

methods (Access = {?nnLayer, ?neuralNetwork})
    % numeric
    function [r, obj] = evaluateNumeric(obj, input, options)
        r = tanh(input);
    end
end

methods
    function [coeffs, d] = computeApproxError(obj, l, u, coeffs)
        order = length(coeffs)-1;
        if order == 1
            % closed form: see [1]

            m = coeffs(1);
            t = coeffs(2);
            
            % compute extreme points of tanh - mx+t; there are two
            % solutions: xu and xl
            xu = atanh(sqrt(1 - m)); % point with max. upper error
            xl = -atanh(sqrt(1 - m)); % point with max. lower error

            % evaluate candidate extreme points within boundary
            xs = [l,xu,xl,u];
            xs = xs(l <= xs & xs <= u);
            ys = obj.f(xs);
            
            % compute approximation error at candidates
            dBounds = ys - (m*xs+t);
            
            % compute approximation error
            du = max(dBounds);
            dl = min(dBounds);
            dc = (du+dl)/2;
            d = du-dc;

            % shift coeffs by center
            coeffs = [m, t + dc];

            computeApproxError@nnActivationLayer(obj,l,u,coeffs);
        else
            % compute in super class
            [coeffs,d] = computeApproxError@nnSShapeLayer(obj,l,u,coeffs);
        end
    end
end

methods(Access=protected)
    function [xs,dxsdm] = computeExtremePointsBatch(obj, m, options)
        % compute extreme points of tanh - mx+t; there are two
        % solutions: xu and xl
        m = max(min(1,m),eps('like',m));
        xu = atanh(sqrt(1 - m)); % point with max. upper error
        xl = -xu; % point with max. lower error
        % list of extreme points
        xs = cat(3,xl,xu);
        % compute derivate wrt. slope m; needed for backpropagation
        dxu = -1./max(2*m.*sqrt(1 - m),eps('like',m));
        dxl = -dxu;
        dxsdm = cat(3,dxl,dxu);
    end
end

end

% ------------------------------ END OF CODE ------------------------------
