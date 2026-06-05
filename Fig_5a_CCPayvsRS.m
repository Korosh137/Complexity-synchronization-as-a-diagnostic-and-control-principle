% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 5a: cooperation and payoff versus sensing radius.
%
% What this script does:
%   - Sweeps the sensing-radius ratio R and estimates final cooperation rate C_r and payoff rate P_r.
%   - Compares adaptive learning against a frozen-threshold baseline.
%   - This script does not compute MDEA/DFA complexity synchronization.
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
ENS    = 10;        % number of ensembles

L      = 2;        % length of the 2D environment
dt     = 1;        % time step
ry     = 21;       % grid size along RS
RS     = 0.24 + 0.01*(1:ry)';   % ratio of vision respect to L
deltaS = 0.1;      % learning step

% ---------------- noise parameters ----------------
EnvSuccessProb = 1;   % probability that a geometrically available fish is actually caught

% ---------------- outputs ----------------
% NOTE: AACS* store ORDINARY correlation (no MDEA-based CS)
AACS1 = zeros(ry, ENS);  % Corr between I1 & T2  (as coded: local_CORRY(1,4))
AACS2 = zeros(ry, ENS);  % Corr between T1 & P2  (as coded: local_CORRY(3,6))
AACS3 = zeros(ry, ENS);  % Corr between I1 & P1  (as coded: local_CORRY(1,5))
AACS4 = zeros(ry, ENS);  % Corr between V1 & V2  (as coded: local_CORRY(7,8))

AACSPvalue1 = nan(ry, ENS);
AACSPvalue2 = nan(ry, ENS);
AACSPvalue3 = nan(ry, ENS);
AACSPvalue4 = nan(ry, ENS);

AACC  = zeros(ry, ENS);   % ratio of mutual cooperation at ti = Time
AAPay = zeros(ry, ENS);   % ratio of average payoff of the S agents at ti = Time

% ---- baseline (frozen thresholds) outputs ----
AACC_base  = zeros(ry, ENS);
AAPay_base = zeros(ry, ENS);

% ========================= PARALLEL + PROGRESS SETUP =========================
if isempty(gcp('nocreate')), parpool; end

% RNG seeds per ensemble
seeds = randi(1e9, ENS, 1);

% Progress bar via DataQueue (~100 ticks per (kk,ens)) for the ADAPTIVE run
chunkSize   = max(1, floor(Time/100));
ticksPerRun = ceil(Time / chunkSize);
totalTicks  = ry * ENS * ticksPerRun;

dq    = parallel.pool.DataQueue;
hWait = [];
try
    hWait = waitbar(0, 'Starting grid (0%)', 'Name', 'Simulation Progress');
    setappdata(hWait, 'count', 0);
    setappdata(hWait, 'total', totalTicks);
    afterEach(dq, @(~) localIncrement(hWait));
catch
    hWait = [];
    disp('Progress: [text mode] ...');
    afterEach(dq, @(~) localTextProgress());
end
% ============================================================================

% ============================= PARFOR OVER RY =============================
parfor kk = 1:ry
    AACS_row1 = zeros(1, ENS);
    AACS_row2 = zeros(1, ENS);
    AACS_row3 = zeros(1, ENS);
    AACS_row4 = zeros(1, ENS);

    AACSPvalue_row1 = zeros(1, ENS);
    AACSPvalue_row2 = zeros(1, ENS);
    AACSPvalue_row3 = zeros(1, ENS);
    AACSPvalue_row4 = zeros(1, ENS);

    AACC_row       = zeros(1, ENS);
    AAPay_row      = zeros(1, ENS);
    AACC_base_row  = zeros(1, ENS);
    AAPay_base_row = zeros(1, ENS);

    for ens = 1:ENS
        % Use same seed for BOTH adaptive + baseline so they are comparable
        baseSeed = seeds(ens) + kk;

        % ----------- per-kk parameters -----------
        rSfac  = RS(kk);

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

        % time series of velocity x-components (for correlation)
        IndividualS1Vx = zeros(Time, 1);
        IndividualS2Vx = zeros(Time, 1);
        IndividualFVx  = zeros(Time, 1);

        xF = L*rand(1, 1);  yF = L*rand(1, 1);
        xS = L*rand(1, 2);  yS = L*rand(1, 2);
        xFS = zeros(1, 3);
        yFS = zeros(1, 3);

        Ratio_CC  = zeros(Time, 1);  CC = 0;
        Ratio_Pay = zeros(Time, 1);  TotalPay = 0;

        % ----- initial velocities at ti = 1 -----
        IndividualS1Vx(1) = velS * cos(thetaS(1));
        IndividualS2Vx(1) = velS * cos(thetaS(2));
        IndividualFVx(1)  = velF * cos(thetaF(1));

        % ===== MAIN TIME LOOP (adaptive) =====
        for ti = 2:Time
            % Pack state vectors
            xFS(1) = xF(1);     yFS(1) = yF(1);
            xFS(2) = xS(1, 1);  yFS(2) = yS(1, 1);
            xFS(3) = xS(1, 2);  yFS(3) = yS(1, 2);
            thetaFS(1) = thetaF(1);
            thetaFS(2) = thetaS(1);
            thetaFS(3) = thetaS(2);

            % -------------- F angles --------------
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

            % -------------- S angles --------------
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

            % -------------- SHARE INFO --------------
            ShaInfo1 = 0;  if rand > ShareInfoS1(ti), ShaInfo1 = 1; end
            ShaInfo2 = 0;  if rand > ShareInfoS2(ti), ShaInfo2 = 1; end

            % -------------- TRUST --------------
            Trusted1  = 0;
            if rand > TrustS1(ti), Trusted1 = 1; thetaS(1) = thetaFoeSShared(2); end
            Trusted2  = 0;
            if rand > TrustS2(ti), Trusted2 = 1; thetaS(2) = thetaFoeSShared(1); end

            % Integrate + wrap (periodic torus)
            xF = mod(xF + velF * cos(thetaF) * dt, L);
            yF = mod(yF + velF * sin(thetaF) * dt, L);
            xS = mod(xS + velS * cos(thetaS) * dt, L);
            yS = mod(yS + velS * sin(thetaS) * dt, L);

            % -------------- PAYOFF neighborhood --------------
            xFS = [xF(1), xS(1, 1), xS(1, 2)];
            yFS = [yF(1), yS(1, 1), yS(1, 2)];
            [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rG, L);

            % --------- ENVIRONMENTAL NOISE: probabilistic catching of F ---------
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
            % --------------------------------------------------------------------

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

            IndividualS1Vx(ti) = velS * cos(thetaS(1));
            IndividualS2Vx(ti) = velS * cos(thetaS(2));
            IndividualFVx(ti)  = velF * cos(thetaF(1));

            % === PROGRESS TICK (adaptive only) ===
            if (mod(ti, chunkSize) == 0) || (ti == Time)
                send(dq, 1);
            end
        end % time (adaptive)

        % ================== Correlation stats (single kk,ens) ==================
        ST0 = floor(0.25 * length(SharePayS1));
        EN0 = length(SharePayS1) - 1;

        data = zeros(EN0 - ST0 + 1, 8);
        data(:, 1) = ShareInfoS1( ST0:EN0);
        data(:, 2) = ShareInfoS2( ST0:EN0);
        data(:, 3) = TrustS1(     ST0:EN0);
        data(:, 4) = TrustS2(     ST0:EN0);
        data(:, 5) = SharePayS1(  ST0:EN0);
        data(:, 6) = SharePayS2(  ST0:EN0);
        data(:, 7) = IndividualS1Vx(ST0:EN0);
        data(:, 8) = IndividualS2Vx(ST0:EN0);

        local_CORRY  = zeros(8, 8);
        local_CORRYp = ones(8, 8);

        for i = 1:8
            for j = 1:8
                x1 = data(1:end-1, i);
                x2 = data(1:end-1, j);

                % Keep your original "diff for i>=6" logic as you had it
                if i >= 6, x1 = diff(data(:, i)); end
                if j >= 6, x2 = diff(data(:, j)); end

                [a1, p1] = corrcoef(x1, x2);
                local_CORRY(i, j)  = a1(2, 1);
                local_CORRYp(i, j) = p1(2, 1);
            end
        end

        AACS_row1(ens)       = local_CORRY(1, 4);
        AACS_row2(ens)       = local_CORRY(3, 6);
        AACS_row3(ens)       = local_CORRY(1, 5);
        AACS_row4(ens)       = local_CORRY(7, 8);

        AACSPvalue_row1(ens) = local_CORRYp(1, 4);
        AACSPvalue_row2(ens) = local_CORRYp(3, 6);
        AACSPvalue_row3(ens) = local_CORRYp(1, 5);
        AACSPvalue_row4(ens) = local_CORRYp(7, 8);

        AACC_row(ens)  = Ratio_CC(end);
        AAPay_row(ens) = Ratio_Pay(end);

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

        Ratio_CC0  = zeros(Time, 1);  CC0 = 0;
        Ratio_Pay0 = zeros(Time, 1);  TotalPay0 = 0;

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

            ShaInfo1 = 0;  if rand > ShareInfoS1(ti), ShaInfo1 = 1; end
            ShaInfo2 = 0;  if rand > ShareInfoS2(ti), ShaInfo2 = 1; end

            Trusted1  = 0;
            if rand > TrustS1(ti), Trusted1 = 1; thetaS(1) = thetaFoeSShared(2); end
            Trusted2  = 0;
            if rand > TrustS2(ti), Trusted2 = 1; thetaS(2) = thetaFoeSShared(1); end

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

            if zPayS1 == -13, PayS(1) = -2; else, PayS(1) = 2; end
            if zPayS2 == -13, PayS(2) = -2; else, PayS(2) = 2; end

            any_win = ((zPayS1 == 13) || (zPayS2 == 13));

            s1 = (rand > SharePayS1(ti));
            s2 = (rand > SharePayS2(ti));

            if Trusted1 == 1 || Trusted2 == 1
                if any_win
                    if s1 && s2
                        PayS = [2, 2];  CC0 = CC0 + 1;
                    end
                    if ~s1 && s2, PayS = [2 + tc, -2]; end
                    if s1 && ~s2, PayS = [-2, 2 + tc]; end
                    if ~s1 && ~s2, PayS = [-1, -1]; end
                end
            end

            % ========= FREEZE thresholds via Used=0 in ALL UpdateThreshold calls =========
            ShareInfoS1(ti + 1) = UpdateThreshold(0, ShareInfoS1(ti), ShaInfo1,  PayS(1), PayS0(1), deltaS);
            ShareInfoS2(ti + 1) = UpdateThreshold(0, ShareInfoS2(ti), ShaInfo2,  PayS(2), PayS0(2), deltaS);
            TrustS1(ti + 1)     = UpdateThreshold(0, TrustS1(ti),     Trusted1,  PayS(1), PayS0(1), deltaS);
            TrustS2(ti + 1)     = UpdateThreshold(0, TrustS2(ti),     Trusted2,  PayS(2), PayS0(2), deltaS);
            SharePayS1(ti + 1)  = UpdateThreshold(0, SharePayS1(ti),  s1,        PayS(1), PayS0(1), deltaS);
            SharePayS2(ti + 1)  = UpdateThreshold(0, SharePayS2(ti),  s2,        PayS(2), PayS0(2), deltaS);

            PayS0 = PayS;
            TotalPay0      = TotalPay0 + mean(PayS);
            Ratio_Pay0(ti) = TotalPay0 / ti;
            Ratio_CC0(ti)  = CC0 / ti;
        end % time (baseline)

        AACC_base_row(ens)  = Ratio_CC0(end);
        AAPay_base_row(ens) = Ratio_Pay0(end);
    end % ens

    % ---------- commit kk-th row ----------
    AACS1(kk, :)       = AACS_row1;
    AACS2(kk, :)       = AACS_row2;
    AACS3(kk, :)       = AACS_row3;
    AACS4(kk, :)       = AACS_row4;

    AACSPvalue1(kk, :) = AACSPvalue_row1;
    AACSPvalue2(kk, :) = AACSPvalue_row2;
    AACSPvalue3(kk, :) = AACSPvalue_row3;
    AACSPvalue4(kk, :) = AACSPvalue_row4;

    AACC(kk, :)        = AACC_row;
    AAPay(kk, :)       = AAPay_row;

    AACC_base(kk, :)   = AACC_base_row;
    AAPay_base(kk, :)  = AAPay_base_row;
end % parfor kk

% Close progress bar
try
    if ~isempty(hWait) && isvalid(hWait)
        waitbar(1, hWait, 'Done (100%)');  pause(0.05);
        close(hWait);
    end
catch
end

% ===================== Performance & CC vs RS (error bars) =====================

Perf_mean      = mean(AAPay,      2, 'omitnan');
Perf_base_mean = mean(AAPay_base, 2, 'omitnan');
CC_mean        = mean(AACC,       2, 'omitnan');

if ENS > 1
    Perf_err      = std(AAPay,      0, 2, 'omitnan');
    Perf_base_err = std(AAPay_base, 0, 2, 'omitnan');
    CC_err        = std(AACC,       0, 2, 'omitnan');
else
    Perf_err      = zeros(size(Perf_mean));
    Perf_base_err = zeros(size(Perf_base_mean));
    CC_err        = zeros(size(CC_mean));
end

% ===================== Figure =====================
fig = figure('Units','pixels','Position',[140 140 980 520]);
ax  = axes(fig);
hold(ax,'on'); box(ax,'on');
grid(ax,'off');                     % NO guide lines

% GLOBAL STYLE (bold ticks too)
set(ax,'FontName','Arial','FontSize',14,'LineWidth',1.2,'FontWeight','bold');

% ---------------- LEFT axis (MC) ----------------
CC_lo = min(CC_mean - CC_err, [], 'omitnan');
CC_hi = max(CC_mean + CC_err, [], 'omitnan');
if ~isfinite(CC_lo), CC_lo = 0; end
if ~isfinite(CC_hi), CC_hi = 1; end
CC_pad = 0.08 * (CC_hi - CC_lo + eps);
CC_lo  = CC_lo - CC_pad;
CC_hi  = CC_hi + CC_pad;

CC_rng = CC_hi - CC_lo;
if CC_rng <= 0.8
    CC_step = 0.1;
elseif CC_rng <= 2
    CC_step = 0.2;
elseif CC_rng <= 4
    CC_step = 0.5;
elseif CC_rng <= 8
    CC_step = 1;
else
    CC_step = 2;
end
CC_ymin = CC_step * floor(CC_lo / CC_step);
CC_ymax = CC_step * ceil (CC_hi / CC_step);

yyaxis(ax,'left')

% Markers + errorbars ONLY (NO connecting line through markers)
mszCC = 6;
h1 = errorbar(ax, RS, CC_mean, CC_err, ...
    'LineStyle','none','Marker','s','MarkerSize',mszCC, ...
    'LineWidth',1.6);

ylabel(ax,'Cooperation rate (Cr)', ...
    'FontName','Arial','FontSize',16,'FontWeight','bold');

ylim(ax,[CC_ymin CC_ymax]);
yticks(ax,CC_ymin:CC_step:CC_ymax);
set(ax,'YColor','k','FontWeight','bold');

% Draw gapped connecting line (does NOT go through markers)
ccCol = h1.Color;
plot_gapped_line(ax, RS, CC_mean, '--', ccCol, 1.6, mszCC);

% ---------------- RIGHT axis (Performance) ----------------
P_lo = min([Perf_mean-Perf_err; Perf_base_mean-Perf_base_err], [], 'omitnan');
P_hi = max([Perf_mean+Perf_err; Perf_base_mean+Perf_base_err], [], 'omitnan');
if ~isfinite(P_lo), P_lo = 0; end
if ~isfinite(P_hi), P_hi = 1; end
P_pad = 0.08 * (P_hi - P_lo + eps);
P_lo  = P_lo - P_pad;
P_hi  = P_hi + P_pad;
P_lo  = min(P_lo,0);

P_rng = P_hi - P_lo;
if P_rng <= 0.4
    P_step = 0.05;
elseif P_rng <= 0.8
    P_step = 0.1;
elseif P_rng <= 1.6
    P_step = 0.2;
elseif P_rng <= 3
    P_step = 0.5;
else
    P_step = 1;
end
P_ymin = P_step * floor(P_lo / P_step);
P_ymax = P_step * ceil (P_hi / P_step);

yyaxis(ax,'right')

mszP = 6;
h2 = errorbar(ax, RS, Perf_mean, Perf_err, ...
    'LineStyle','none','Marker','o','MarkerSize',mszP, ...
    'LineWidth',1.6);

% ---- FORCE baseline (learning inactive / non interaction) to BLACK ----
h3 = errorbar(ax, RS, Perf_base_mean, Perf_base_err, ...
    'LineStyle','none','Marker','d','MarkerSize',mszP, ...
    'LineWidth',1.6, 'Color','k', 'MarkerEdgeColor','k', 'MarkerFaceColor','k');

ylabel(ax,'Payoff rate (Pr)', ...
    'FontName','Arial','FontSize',16,'FontWeight','bold');

ylim(ax,[P_ymin P_ymax]);
yticks(ax,P_ymin:P_step:P_ymax);
set(ax,'YColor','k','FontWeight','bold');

% Draw gapped connecting lines (do NOT go through markers)
pCol  = h2.Color;
pbCol = [0 0 0]; % BLACK for baseline
plot_gapped_line(ax, RS, Perf_mean,      '-', pCol,  1.6, mszP);
plot_gapped_line(ax, RS, Perf_base_mean, ':', pbCol, 1.6, mszP);

% ---------------- Common ----------------
xlabel(ax,'R', 'FontName','Arial','FontSize',16,'FontWeight','bold');
set(ax,'FontWeight','bold'); % ensure x ticks bold too

% ===================== Legend with line+marker samples =====================
% Create proxy (dummy) handles so legend shows BOTH line style and marker icon
% (instead of errorbar caps only).
hL1 = plot(ax, nan, nan, '--s', 'Color', ccCol, 'LineWidth',1.6, ...
    'MarkerSize', mszCC, 'MarkerFaceColor', ccCol, 'MarkerEdgeColor', ccCol);
hL2 = plot(ax, nan, nan, '-o',  'Color', pCol,  'LineWidth',1.6, ...
    'MarkerSize', mszP, 'MarkerFaceColor', pCol, 'MarkerEdgeColor', pCol);
hL3 = plot(ax, nan, nan, ':d',  'Color', [0 0 0], 'LineWidth',1.6, ...
    'MarkerSize', mszP, 'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', [0 0 0]);

lg = legend(ax,[hL1 hL2 hL3], ...
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
            waitbar(frac, hWait, sprintf('Running grid (%.1f%%%%)', 100*frac));
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
        aa = pi0 +  1e-3*randn;
    end
    aa = min(max(aa, 0), 1);
end

% === Draw a polyline with a small GAP around every marker (pixel-accurate) ===
function plot_gapped_line(ax, x, y, lineStyle, lineColor, lw, markerSizePts)
    x = x(:); y = y(:);
    ok = isfinite(x) & isfinite(y);
    x = x(ok); y = y(ok);
    if numel(x) < 2, return; end

    % gap in pixels from marker size (points -> pixels)
    ppi = get(0,'ScreenPixelsPerInch');           % pixels / inch
    ms_px = (markerSizePts * ppi) / 72;          % points -> pixels
    gapPix = 0.65 * ms_px;                       % tuned so line clears marker face

    % axis pixel geometry
    axPix = getpixelposition(ax, true);
    W = axPix(3); H = axPix(4);

    % current limits for mapping
    xl = xlim(ax);
    yl = ylim(ax);

    % helper: data->pixel
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

        f = gapPix / Lp;   % fraction of segment to trim at each end
        xa = x1 + f*(x2-x1);
        ya = y1 + f*(y2-y1);
        xb = x2 - f*(x2-x1);
        yb = y2 - f*(y2-y1);

        plot(ax, [xa xb], [ya yb], ...
            'LineStyle', lineStyle, 'Color', lineColor, 'LineWidth', lw);
    end
end
