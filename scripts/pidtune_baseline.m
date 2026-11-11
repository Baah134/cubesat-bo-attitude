% MATLAB pidtune baseline (same role as in the IECON 2026 paper).
% Per-axis plant: torque command -> attitude angle
%   G(s) = 1 / (J s^2 (T_m s + 1)) * exp(-s*Ts/2)
% where exp(-s*Ts/2) approximates the 10 Hz zero-order hold.
% pidtune's default design: 60 deg phase margin, crossover picked by pidtune.
% It does not see torque/momentum limits (that is the point of the comparison).
% Output: configs/pidtune_gains.json  (read by optim/pidtune.py)
%
% Run:  matlab -batch "run('scripts/pidtune_baseline.m')"   from the repo root

J_axes = [0.042, 0.0067];     % x/y (transverse) and z
names  = {'xy', 'z'};
T_m = 0.03; Ts = 0.1;
out = struct();
out.matlab_version = version;
out.method = 'pidtune(G, ''PID'') default options, continuous plant with Ts/2 input delay';
for i = 1:numel(J_axes)
    J = J_axes(i);
    s = tf('s');
    G = 1 / (J * s^2 * (T_m * s + 1));
    G.InputDelay = Ts / 2;
    [C, info] = pidtune(G, 'PID');
    r = struct('J', J, 'Kp', C.Kp, 'Ki', C.Ki, 'Kd', C.Kd, ...
               'crossover_rad_s', info.CrossoverFrequency, 'phase_margin_deg', info.PhaseMargin, ...
               'stable', info.Stable);
    out.(names{i}) = r;
    fprintf('%s: Kp=%.4g Ki=%.4g Kd=%.4g wc=%.3g rad/s PM=%.1f deg\n', names{i}, C.Kp, C.Ki, C.Kd, ...
            info.CrossoverFrequency, info.PhaseMargin);
end
repo = fileparts(fileparts(mfilename('fullpath')));
fid = fopen(fullfile(repo, 'configs', 'pidtune_gains.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
