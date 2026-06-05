% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 5b: cooperation and payoff time series.
%
% What this script does:
%   - Runs simulations at one sensing-radius condition and plots ensemble-averaged C_r(t) and P_r(t).
%   - Includes the learning-active payoff curve and the frozen-threshold baseline payoff curve.
%   - This script is for performance validation and does not compute CS.
%   - Parallel Computing Toolbox is used if available; ensemble seeds are generated at run time.
%
% Model variables:
%   I = information-sharing threshold, T = trust threshold, P = payoff-sharing threshold.
%   Subscripts 1 and 2 denote the two predator agents. F denotes the prey.
%
% Notes for users:
%   - Large final runs can be slow because most scripts use Time = 1e6 and multiple ensembles.
%   - Several scripts use local helper functions defined at the end of the same file.
%   - If a parallel pool cannot be opened, run with fewer ensembles or replace PARFOR by FOR.
%   - Random-number behavior is documented above for each script; set rng(...) manually if exact
%     reproducibility is required.
%
% -------------------------------------------------------------------------
tic
clc
clear all
close all

Time   = 1e6;      % number of trials
ENS    = 10;       % number of ensembles

L      = 2;        % length of the 2D environment
dt     = 1;        % time step
RS_index = 1;                  % <<< pick which RS to show (1..ry)
RS     = 0.45 ; % RSgrid(RS_index);     % single RS used for the time series
deltaS = 0.1;      % learning step

% ---------------- noise parameters ----------------
EnvSuccessProb = 1;   % probability that a geometrically available fish is actually caught

% ========================= PARALLEL + PROGRESS SETUP =========================
if isempty(gcp('nocreate')), parpool; end

% RNG seeds per ensemble
seeds = randi(1e9, ENS, 1);

% Progress bar via DataQueue (~200 ticks per ensemble) for the ADAPTIVE run
chunkSize   = max(1, floor(Time/200));
ticksPerRun = ceil(Time / chunkSize);
totalTicks  = ENS * ticksPerRun;

dq    = parallel.pool.DataQueue;
hWait = [];
try
    hWait = waitbar(0, 'Starting (0%)', 'Name', 'Simulation Progress');
    setappdata(hWait, 'count', 0);
    setappdata(hWait, 'total', totalTicks);
    afterEach(dq, @(~) localIncrement(hWait));
catch
    hWait = [];
    disp('Progress: [text mode] ...');
    afterEach(dq, @(~) localTextProgress());
end
% ============================================================================

% ========================= STORE TIME SERIES (PER ENS) =========================
CC_ts      = zeros(Time, ENS);
Pay_ts     = zeros(Time, ENS);
Pay0_ts    = zeros(Time, ENS);

% ============================= PARFOR OVER ENS =============================
parfor ens = 1:ENS
    baseSeed = seeds(ens) + RS_index;  % stable across RS_index choices
    rSfac    = RS;

    % ======================= ADAPTIVE RUN =======================
    rng(baseSeed, 'twister');

    ThetaDecept = pi/6;

    % S/F params
    rS   = rSfac * L;   % radius of vision of S agents
    velS = 0.1;         % speed of S agents
    rF   = 1 * rS;      % radius of vision of F agent
    velF = 2 * velS;    % speed of F agent
    rG   = 1 * rS;      % radius of game play
    tc   = 3;           % temptation to cheat in the PD game

    % Initial conditions
    thetaF           = 2*pi*(rand(1, 1) - 0.5);  % angles (one F)
    thetaFoeF        = zeros(1, 1);
    thetaFoeFDECEPT  = zeros(1, 1);
    thetaS           = 2*pi*(rand(1, 2) - 0.5);  % angles (two S)
    thetaFS          = 2*pi*(rand(1, 3) - 0.5);
    thetaFoeSShared  = zeros(1, 2);

    PayS0 = zeros(2, 1);  PayS = zeros(2, 1);

    ShareInfoS1 = ones(Time, 1);
    ShareInfoS2 = ones(Time, 1);
    TrustS1     = ones(Time, 1);
    TrustS2     = ones(Time, 1);
    SharePayS1  = ones(Time, 1);
    SharePayS2  = ones(Time, 1);

    xF = L*rand(1, 1);  yF = L*rand(1, 1);
    xS = L*rand(1, 2);  yS = L*rand(1, 2);
    xFS = zeros(1, 3);
    yFS = zeros(1, 3);

    Ratio_CC  = zeros(Time, 1);  CC = 0;
    Ratio_Pay = zeros(Time, 1);  TotalPay = 0;

    % initial values at ti=1
    Ratio_CC(1)  = 0;
    Ratio_Pay(1) = 0;

    for ti = 2:Time
        % Pack state vectors
        xFS(1) = xF(1);     yFS(1) = yF(1);
        xFS(2) = xS(1, 1);  yFS(2) = yS(1, 1);
        xFS(3) = xS(1, 2);  yFS(3) = yS(1, 2);
        thetaFS(1) = thetaF(1);
        thetaFS(2) = thetaS(1);
        thetaFS(3) = thetaS(2);

        % -------- F angles --------
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rF, L);
        list = l1FS(l2FS == 1);  list = list(list > 1);
        if ~isempty(list)
            xSthatFpredicts = mean(xFS(list)) + velS*mean(cos(thetaFS(list)))*dt;
            ySthatFpredicts = mean(yFS(list)) + velS*mean(sin(thetaFS(list)))*dt;
            tet1            = AnglePeriodic_torus(xSthatFpredicts, ySthatFpredicts, xFS(1), yFS(1), L);
            thetaDeflect    = (ThetaDecept) * sign(-1 + 2*rand);
            thetaFoeF(1)        = pi + tet1 + thetaDeflect;
            thetaFoeFDECEPT(1)  = pi + tet1 - thetaDeflect;
        else
            thetaFoeF(1)        = thetaFS(1);
            thetaFoeFDECEPT(1)  = thetaFS(1);
        end
        if rand < 0.5
            thetaF(1) = thetaFoeF(1);
        else
            thetaF(1) = thetaFoeFDECEPT(1);
        end

        % -------- S angles --------
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rS, L);

        % S1 block
        list = l1FS(l2FS == 2);  list = list(list <= 1);
        if ~isempty(list)
            xFthatSpredicts        = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
            yFthatSpredicts        = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
            xFthatSpredictsDECEPT  = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
            yFthatSpredictsDECEPT  = mean(yFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;
            if rand < 0.5
                thetaS(1)          = AnglePeriodic_torus(xFthatSpredicts,       yFthatSpredicts,       xFS(2), yFS(2), L);
                thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(3), yFS(3), L);
            else
                thetaS(1)          = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(2), yFS(2), L);
                thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredicts,       yFthatSpredicts,       xFS(3), yFS(3), L);
            end
        else
            thetaS(1) = thetaFS(2);
        end

        % S2 block
        list = l1FS(l2FS == 3);  list = list(list <= 1);
        if ~isempty(list)
            xFthatSpredicts        = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
            yFthatSpredicts        = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
            xFthatSpredictsDECEPT  = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
            yFthatSpredictsDECEPT  = mean(yFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;
            if rand < 0.5
                thetaS(2)          = AnglePeriodic_torus(xFthatSpredicts,       yFthatSpredicts,       xFS(3), yFS(3), L);
                thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(2), yFS(2), L);
            else
                thetaS(2)          = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(3), yFS(3), L);
                thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredicts,       yFthatSpredicts,       xFS(2), yFS(2), L);
            end
        else
            thetaS(2) = thetaFS(3);
        end

        % -------- decisions --------
        ShaInfo1 = 0;  if rand > ShareInfoS1(ti), ShaInfo1 = 1; end
        ShaInfo2 = 0;  if rand > ShareInfoS2(ti), ShaInfo2 = 1; end

        Trusted1  = 0;
        if rand > TrustS1(ti), Trusted1 = 1; thetaS(1) = thetaFoeSShared(2); end
        Trusted2  = 0;
        if rand > TrustS2(ti), Trusted2 = 1; thetaS(2) = thetaFoeSShared(1); end

        % Integrate + wrap
        xF = mod(xF + velF * cos(thetaF) * dt, L);
        yF = mod(yF + velF * sin(thetaF) * dt, L);
        xS = mod(xS + velS * cos(thetaS) * dt, L);
        yS = mod(yS + velS * sin(thetaS) * dt, L);

        % payoff neighborhood
        xFS = [xF(1), xS(1, 1), xS(1, 2)];
        yFS = [yF(1), yS(1, 1), yS(1, 2)];
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rG, L);

        list = l1FS(l2FS == 2);  listofF = list(list <= 1);  LF1 = length(listofF);
        list = l1FS(l2FS == 3);  listofF = list(list <= 1);  LF2 = length(listofF);

        if LF1 == 0
            zPayS1 = -13;
        else
            if rand < EnvSuccessProb, zPayS1 = 13 * LF1; else, zPayS1 = -13; end
        end

        if LF2 == 0
            zPayS2 = -13;
        else
            if rand < EnvSuccessProb, zPayS2 = 13 * LF2; else, zPayS2 = -13; end
        end

        if zPayS1 == -13, PayS(1) = -2; else, PayS(1) = 2; end
        if zPayS2 == -13, PayS(2) = -2; else, PayS(2) = 2; end

        any_win = ((zPayS1 == 13) || (zPayS2 == 13));

        s1 = (rand > SharePayS1(ti));
        s2 = (rand > SharePayS2(ti));

        ShareUSedS1 = 0;  ShareUSedS2 = 0;
        if Trusted1 == 1 || Trusted2 == 1
            if any_win
                ShareUSedS1 = 1;  ShareUSedS2 = 1;

                if s1 && s2
                    PayS = [2, 2];  CC = CC + 1;
                end
                if ~s1 && s2
                    PayS = [2 + tc, -2];
                end
                if s1 && ~s2
                    PayS = [-2, 2 + tc];
                end
                if ~s1 && ~s2
                    PayS = [-1, -1];
                end
            end
        end

        ShareInfoS1(ti + 1) = UpdateThreshold(1, ShareInfoS1(ti),  ShaInfo1,  PayS(1), PayS0(1), deltaS);
        ShareInfoS2(ti + 1) = UpdateThreshold(1, ShareInfoS2(ti),  ShaInfo2,  PayS(2), PayS0(2), deltaS);
        TrustS1(ti + 1)     = UpdateThreshold(1, TrustS1(ti),      Trusted1,  PayS(1), PayS0(1), deltaS);
        TrustS2(ti + 1)     = UpdateThreshold(1, TrustS2(ti),      Trusted2,  PayS(2), PayS0(2), deltaS);
        SharePayS1(ti + 1)  = UpdateThreshold(ShareUSedS1, SharePayS1(ti),   s1,       PayS(1), PayS0(1), 1*deltaS);
        SharePayS2(ti + 1)  = UpdateThreshold(ShareUSedS2, SharePayS2(ti),   s2,       PayS(2), PayS0(2), deltaS);

        PayS0 = PayS;
        TotalPay      = TotalPay + mean(PayS);
        Ratio_Pay(ti) = TotalPay / ti;
        Ratio_CC(ti)  = CC / ti;

        if (mod(ti, chunkSize) == 0) || (ti == Time)
            send(dq, 1);
        end
    end % time (adaptive)

    % ======================= BASELINE RUN (THRESHOLDS FROZEN) =======================
    rng(baseSeed, 'twister');  % same randomness

    ThetaDecept = pi/6;

    rS   = rSfac * L;
    velS = 0.1;
    rF   = 1 * rS;
    velF = 2 * velS;
    rG   = 1 * rS;
    tc   = 3;

    thetaF           = 2*pi*(rand(1, 1) - 0.5);
    thetaFoeF        = zeros(1, 1);
    thetaFoeFDECEPT  = zeros(1, 1);
    thetaS           = 2*pi*(rand(1, 2) - 0.5);
    thetaFS          = 2*pi*(rand(1, 3) - 0.5);
    thetaFoeSShared  = zeros(1, 2);

    PayS0b = zeros(2, 1);  PaySb = zeros(2, 1);

    ShareInfoS1b = ones(Time, 1);
    ShareInfoS2b = ones(Time, 1);
    TrustS1b     = ones(Time, 1);
    TrustS2b     = ones(Time, 1);
    SharePayS1b  = ones(Time, 1);
    SharePayS2b  = ones(Time, 1);

    xF = L*rand(1, 1);  yF = L*rand(1, 1);
    xS = L*rand(1, 2);  yS = L*rand(1, 2);
    xFS = zeros(1, 3);
    yFS = zeros(1, 3);

    Ratio_CC0  = zeros(Time, 1);  CC0 = 0;
    Ratio_Pay0 = zeros(Time, 1);  TotalPay0 = 0;

    Ratio_CC0(1)  = 0;
    Ratio_Pay0(1) = 0;

    for ti = 2:Time
        xFS(1) = xF(1);     yFS(1) = yF(1);
        xFS(2) = xS(1, 1);  yFS(2) = yS(1, 1);
        xFS(3) = xS(1, 2);  yFS(3) = yS(1, 2);
        thetaFS(1) = thetaF(1);
        thetaFS(2) = thetaS(1);
        thetaFS(3) = thetaS(2);

        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rF, L);
        list = l1FS(l2FS == 1);  list = list(list > 1);
        if ~isempty(list)
            xSthatFpredicts = mean(xFS(list)) + velS*mean(cos(thetaFS(list)))*dt;
            ySthatFpredicts = mean(yFS(list)) + velS*mean(sin(thetaFS(list)))*dt;
            tet1            = AnglePeriodic_torus(xSthatFpredicts, ySthatFpredicts, xFS(1), yFS(1), L);
            thetaDeflect    = (ThetaDecept) * sign(-1 + 2*rand);
            thetaFoeF(1)        = pi + tet1 + thetaDeflect;
            thetaFoeFDECEPT(1)  = pi + tet1 - thetaDeflect;
        else
            thetaFoeF(1)        = thetaFS(1);
            thetaFoeFDECEPT(1)  = thetaFS(1);
        end
        if rand < 0.5, thetaF(1) = thetaFoeF(1); else, thetaF(1) = thetaFoeFDECEPT(1); end

        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rS, L);

        list = l1FS(l2FS == 2);  list = list(list <= 1);
        if ~isempty(list)
            xFthatSpredicts        = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
            yFthatSpredicts        = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
            xFthatSpredictsDECEPT  = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
            yFthatSpredictsDECEPT  = mean(xFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;
            if rand < 0.5
                thetaS(1)          = AnglePeriodic_torus(xFthatSpredicts,       yFthatSpredicts,       xFS(2), yFS(2), L);
                thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(3), yFS(3), L);
            else
                thetaS(1)          = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(2), yFS(2), L);
                thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredicts,       yFthatSpredicts,       xFS(3), yFS(3), L);
            end
        else
            thetaS(1) = thetaFS(2);
        end

        list = l1FS(l2FS == 3);  list = list(list <= 1);
        if ~isempty(list)
            xFthatSpredicts        = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
            yFthatSpredicts        = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
            xFthatSpredictsDECEPT  = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
            yFthatSpredictsDECEPT  = mean(xFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;
            if rand < 0.5
                thetaS(2)          = AnglePeriodic_torus(xFthatSpredicts,       yFthatSpredicts,       xFS(3), yFS(3), L);
                thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(2), yFS(2), L);
            else
                thetaS(2)          = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(3), yFS(3), L);
                thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredicts,       yFthatSpredicts,       xFS(2), yFS(2), L);
            end
        else
            thetaS(2) = thetaFS(3);
        end

        ShaInfo1 = 0;  if rand > ShareInfoS1b(ti), ShaInfo1 = 1; end
        ShaInfo2 = 0;  if rand > ShareInfoS2b(ti), ShaInfo2 = 1; end

        Trusted1  = 0;
        if rand > TrustS1b(ti), Trusted1 = 1; thetaS(1) = thetaFoeSShared(2); end
        Trusted2  = 0;
        if rand > TrustS2b(ti), Trusted2 = 1; thetaS(2) = thetaFoeSShared(1); end

        xF = mod(xF + velF * cos(thetaF) * dt, L);
        yF = mod(yF + velF * sin(thetaF) * dt, L);
        xS = mod(xS + velS * cos(thetaS) * dt, L);
        yS = mod(yS + velS * sin(thetaS) * dt, L);

        xFS = [xF(1), xS(1, 1), xS(1, 2)];
        yFS = [yF(1), yS(1, 1), yS(1, 2)];
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rG, L);

        list = l1FS(l2FS == 2);  listofF = list(list <= 1);  LF1 = length(listofF);
        list = l1FS(l2FS == 3);  listofF = list(list <= 1);  LF2 = length(listofF);

        if LF1 == 0
            zPayS1 = -13;
        else
            if rand < EnvSuccessProb, zPayS1 = 13 * LF1; else, zPayS1 = -13; end
        end

        if LF2 == 0
            zPayS2 = -13;
        else
            if rand < EnvSuccessProb, zPayS2 = 13 * LF2; else, zPayS2 = -13; end
        end

        if zPayS1 == -13, PaySb(1) = -2; else, PaySb(1) = 2; end
        if zPayS2 == -13, PaySb(2) = -2; else, PaySb(2) = 2; end

        any_win = ((zPayS1 == 13) || (zPayS2 == 13));

        s1 = (rand > SharePayS1b(ti));
        s2 = (rand > SharePayS2b(ti));

        if Trusted1 == 1 || Trusted2 == 1
            if any_win
                if s1 && s2
                    PaySb = [2, 2];  CC0 = CC0 + 1;
                end
                if ~s1 && s2, PaySb = [2 + tc, -2]; end
                if s1 && ~s2, PaySb = [-2, 2 + tc]; end
                if ~s1 && ~s2, PaySb = [-1, -1]; end
            end
        end

        % FREEZE thresholds via Used=0 in ALL UpdateThreshold calls
        ShareInfoS1b(ti + 1) = UpdateThreshold(0, ShareInfoS1b(ti), ShaInfo1,  PaySb(1), PayS0b(1), deltaS);
        ShareInfoS2b(ti + 1) = UpdateThreshold(0, ShareInfoS2b(ti), ShaInfo2,  PaySb(2), PayS0b(2), deltaS);
        TrustS1b(ti + 1)     = UpdateThreshold(0, TrustS1b(ti),     Trusted1,  PaySb(1), PayS0b(1), deltaS);
        TrustS2b(ti + 1)     = UpdateThreshold(0, TrustS2b(ti),     Trusted2,  PaySb(2), PayS0b(2), deltaS);
        SharePayS1b(ti + 1)  = UpdateThreshold(0, SharePayS1b(ti),  s1,        PaySb(1), PayS0b(1), deltaS);
        SharePayS2b(ti + 1)  = UpdateThreshold(0, SharePayS2b(ti),  s2,        PaySb(2), PayS0b(2), deltaS);

        PayS0b = PaySb;
        TotalPay0       = TotalPay0 + mean(PaySb);
        Ratio_Pay0(ti)  = TotalPay0 / ti;
        Ratio_CC0(ti)   = CC0 / ti;
    end % time (baseline)

    % store ensemble time series
    CC_ts(:, ens)   = Ratio_CC(:);
    Pay_ts(:, ens)  = Ratio_Pay(:);
    Pay0_ts(:, ens) = Ratio_Pay0(:);
end % parfor ens

% Close progress bar
try
    if ~isempty(hWait) && isvalid(hWait)
        waitbar(1, hWait, 'Done (100%)');  pause(0.05);
        close(hWait);
    end
catch
end

% ===================== AVERAGE ACROSS ENS (time series) =====================
ti_vec = (1:Time)';

CC_mean   = mean(CC_ts,   2, 'omitnan');
Pay_mean  = mean(Pay_ts,  2, 'omitnan');
Pay0_mean = mean(Pay0_ts, 2, 'omitnan');

if ENS > 1
    CC_err   = std(CC_ts,   0, 2, 'omitnan');
    Pay_err  = std(Pay_ts,  0, 2, 'omitnan');
    Pay0_err = std(Pay0_ts, 0, 2, 'omitnan');
else
    CC_err   = zeros(size(CC_mean));
    Pay_err  = zeros(size(Pay_mean));
    Pay0_err = zeros(size(Pay0_mean));
end

% ===================== Figure (vs ti) =====================
fig = figure('Units','pixels','Position',[140 140 980 520]);
ax  = axes(fig);
hold(ax,'on'); box(ax,'on');
grid(ax,'off');

set(ax,'FontName','Arial','FontSize',14,'LineWidth',1.2,'FontWeight','bold');

% ---------- LEFT axis (MCr) ----------
yyaxis(ax,'left')

mszCC = 6;
stride = max(1, floor(Time/250));             % ~250 markers
idxM = 1:stride:Time;

h1m = errorbar(ax, ti_vec(idxM), CC_mean(idxM), CC_err(idxM), ...
    'LineStyle','none','Marker','s','MarkerSize',mszCC,'LineWidth',1.6);

ylabel(ax,'Cooperation rate (Cr)', 'FontName','Arial','FontSize',16,'FontWeight','bold');
set(ax,'YColor','k','FontWeight','bold');

% gapped line (drawn on full resolution)
ccCol = h1m.Color;
plot_gapped_line(ax, ti_vec, CC_mean, '--', ccCol, 1.6, mszCC);

% ---------- RIGHT axis (MPr) ----------
yyaxis(ax,'right')

mszP = 6;

h2m = errorbar(ax, ti_vec(idxM), Pay_mean(idxM), Pay_err(idxM), ...
    'LineStyle','none','Marker','o','MarkerSize',mszP,'LineWidth',1.6);

% FORCE learning-inactive / non-interaction to BLACK
h3m = errorbar(ax, ti_vec(idxM), Pay0_mean(idxM), Pay0_err(idxM), ...
    'LineStyle','none','Marker','d','MarkerSize',mszP,'LineWidth',1.6, ...
    'Color','k','MarkerEdgeColor','k','MarkerFaceColor','k');

ylabel(ax,'Payoff rate (Pr)', 'FontName','Arial','FontSize',16,'FontWeight','bold');
set(ax,'YColor','k','FontWeight','bold');

pCol  = h2m.Color;
pbCol = [0 0 0];
plot_gapped_line(ax, ti_vec, Pay_mean,  '-', pCol,  1.6, mszP);
plot_gapped_line(ax, ti_vec, Pay0_mean, ':', pbCol, 1.6, mszP);

% ---------- Common ----------
xlabel(ax, 't', 'FontName','Arial','FontSize',16,'FontWeight','bold');
set(ax,'FontWeight','bold');

% ===================== Legend: show line + marker (proxy handles) =====================
hL1 = plot(ax, nan, nan, '--s', 'Color', ccCol, 'LineWidth',1.6, ...
    'MarkerSize', mszCC, 'MarkerFaceColor', ccCol, 'MarkerEdgeColor', ccCol);
hL2 = plot(ax, nan, nan, '-o',  'Color', pCol,  'LineWidth',1.6, ...
    'MarkerSize', mszP,  'MarkerFaceColor', pCol, 'MarkerEdgeColor', pCol);
hL3 = plot(ax, nan, nan, ':d',  'Color', [0 0 0], 'LineWidth',1.6, ...
    'MarkerSize', mszP,  'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', [0 0 0]);

lg = legend(ax, [hL1 hL2 hL3], ...
    {'Cr','Pr (learning active)','Pr (learning inactive)'}, ...
    'Location','northwest');
set(lg,'FontSize',14,'FontName','Arial','FontWeight','bold','Box','off');

hold(ax,'off');

toc

% =============================== FUNCTIONS ===============================
function localIncrement(hWait)
    if isempty(hWait) || ~ishandle(hWait), return; end
    try
        count = getappdata(hWait, 'count');
        total = getappdata(hWait, 'total');
        count = count + 1;
        if total <= 0, total = 1; end
        setappdata(hWait, 'count', count);
        frac = min(1, count / total);
        if isvalid(hWait)
            waitbar(frac, hWait, sprintf('Running (%.1f%%%%)', 100*frac));
        end
        drawnow limitrate;
    catch
    end
end

function localTextProgress()
    persistent cnt
    if isempty(cnt), cnt = 0; end
    cnt = cnt + 1;
    if mod(cnt, 1e4) == 0
        fprintf('Progress ticks: %d\n', cnt);
    end
end

function theta = AnglePeriodic_torus(x_to, y_to, x_from, y_from, L)
    dx = x_to - x_from;  dy = y_to - y_from;
    dx = dx - L*floor(dx/L + 0.5);
    dy = dy - L*floor(dy/L + 0.5);
    theta = atan2(dy, dx);
end

function [A, B] = Finddistance_torus(x, y, r, L)
    x = x(:)';  y = y(:)';
    N = numel(x);
    DX = x - x.';
    DY = y - y.';
    DX = DX - L*round(DX./L);
    DY = DY - L*round(DY./L);
    D = hypot(DX, DY);
    D(1:N+1:end) = inf;
    [A, B] = find((D > 0) & (D < r));
end

function aa = UpdateThreshold(Used, pi0, Decision0, Pay, Paybefore, ChangeThreshold)
    if Used == 1
        if Decision0 == 1, dp = -1*ChangeThreshold; else, dp = 1*ChangeThreshold; end
        if Pay ~= 0 && Paybefore ~= 0
            DeltaPay = (Pay - Paybefore);
        else
            DeltaPay = 0;
        end
        DeltaPay = DeltaPay + 1e-3*randn;
        aa = pi0 + dp * DeltaPay;
    else
        aa = pi0 + 1e-3*randn;
    end
    aa = min(max(aa, 0), 1);
end

% === Draw a polyline with a small GAP around every marker (pixel-accurate) ===
function plot_gapped_line(ax, x, y, lineStyle, lineColor, lw, markerSizePts)
    x = x(:); y = y(:);
    ok = isfinite(x) & isfinite(y);
    x = x(ok); y = y(ok);
    if numel(x) < 2, return; end

    ppi = get(0,'ScreenPixelsPerInch');
    ms_px = (markerSizePts * ppi) / 72;
    gapPix = 0.65 * ms_px;

    axPix = getpixelposition(ax, true);
    W = axPix(3); H = axPix(4);

    xl = xlim(ax);
    yl = ylim(ax);

    x2p = @(xx) (xx - xl(1)) ./ max(eps,(xl(2)-xl(1))) * W;
    y2p = @(yy) (yy - yl(1)) ./ max(eps,(yl(2)-yl(1))) * H;

    for k = 1:numel(x)-1
        x1 = x(k);   y1 = y(k);
        x2 = x(k+1); y2 = y(k+1);

        dxp = x2p(x2) - x2p(x1);
        dyp = y2p(y2) - y2p(y1);
        Lp  = hypot(dxp, dyp);

        if ~isfinite(Lp) || Lp <= 2*gapPix
            continue
        end

        f = gapPix / Lp;
        xa = x1 + f*(x2-x1);
        ya = y1 + f*(y2-y1);
        xb = x2 - f*(x2-x1);
        yb = y2 - f*(y2-y1);

        plot(ax, [xa xb], [ya yb], ...
            'LineStyle', lineStyle, 'Color', lineColor, 'LineWidth', lw);
    end
end
