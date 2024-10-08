function [completed,res,tTotal] = example_neuralNet_reach_05_doublePendulum_moreRobust
% example_neuralNet_reach_05_doublePendulum_moreRobust - example of
%    reachability analysis for a neural network controlled double pendulum
%
% Syntax:
%    completed = example_neuralNet_reach_05_doublePendulum_moreRobust()
%
% Inputs:
%    -
%
% Outputs:
%    completed - true/false 
%    res - verification result
%    tTotal - total time
% 
% Reference:
%   [1] Johnson, Taylor T., et al. "ARCH-COMP21 Category Report: 
%       Artificial Intelligence and Neural Network Control Systems (AINNCS)
%       for Continuous and Hybrid Systems Plants." 
%       EPiC Series in Computing 80 (2021): 90-119.

% Authors:       Niklas Kochdumper, Tobias Ladner
% Written:       08-November-2021
% Last update:   23-May-2022 (TL, ARCH'22 revisions)
%                30-March-2023 (TL, verify violated runs, ARCH'23 revisions)
%                02-May-2024 (TL, ARCH'24 revisions)
% Last revision: 14-November-2022 (TL, clean up)

% ------------------------------ BEGIN CODE -------------------------------

disp("BENCHMARK: Double Pendulum (more robust)")

% Parameters --------------------------------------------------------------

R0 = interval([1;1;1;1],[1.3;1.3;1.3;1.3]);

params.tFinal = 0.4;
params.R0 = polyZonotope(R0);

% Reachability Settings ---------------------------------------------------

options.timeStep = 0.02;
options.alg = 'lin';
options.tensorOrder = 2;
options.taylorTerms = 4;
options.zonotopeOrder = 200;

% Options for NN evaluation -----------------------------------------------

options.nn = struct();
options.nn.poly_method = "singh";

% System Dynamics ---------------------------------------------------------

% open-loop system
sys = nonlinearSys(@doublePendulum);

% load neural network controller
% [4, 25, 25, 2]
nn = neuralNetwork.readONNXNetwork('controller_double_pendulum_more_robust.onnx');

% construct neural network controlled system
sys = neurNetContrSys(sys,nn,0.02);

% Specification -----------------------------------------------------------

safeSet = interval([-1.5;-1.5;-1.5;-1.5],[1.5;1.5;1.5;1.5]);
spec = specification(safeSet,'safeSet');

% Verification ------------------------------------------------------------

t = tic;
[res, R, simRes] = verify(sys, spec, params, options, true);
tTotal = toc(t);
disp(['Result: ' res])

% Visualization -----------------------------------------------------------

disp("Plotting..")
figure; hold on; box on;
projDims = [1, 2];

% plot specification
plot(specification(safeSet, 'safeSet'), projDims, 'DisplayName', 'Safe set');

% plot reachable set
useCORAcolors("CORA:contDynamics")
plot(R, projDims, 'DisplayName', 'Reachable set')

% plot initial set
plot(R0, projDims, 'k', 'FaceColor', [1 1 1], 'DisplayName', 'Initial set');

% plot simulations
plot(simRes,projDims, 'DisplayName', 'Simulations');

% labels and legend
xlabel('\theta_1'); ylabel('\theta_2');
legend(Location="northwest")

figure; hold on; box on;
projDims = [3, 4];

% plot specification
plot(specification(safeSet, 'safeSet'), projDims, 'DisplayName', 'Safe set');

% plot reachable set
useCORAcolors("CORA:contDynamics")
plot(R, projDims, 'DisplayName', 'Reachable set')

% plot initial set
plot(R0, projDims, 'k', 'FaceColor', [1 1 1], 'DisplayName', 'Initial set');

% plot simulations
plot(simRes,projDims, 'DisplayName', 'Simulations');

% labels and legend
xlabel('$\dot \theta_1$','interpreter','latex');
ylabel('$\dot \theta_2$','interpreter','latex');
legend(Location="northwest")


% example completed -------------------------------------------------------

completed = true;

% handling for ARCH competition
if nargout < 2
    clear res;
end
if nargout < 3
    clear tTotal;
end

end

% ------------------------------ END OF CODE ------------------------------
