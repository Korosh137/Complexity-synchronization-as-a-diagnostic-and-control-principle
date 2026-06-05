% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 4: representative adaptive threshold trajectories.
%
% What this script does:
%   - Runs one short simulation and plots the six adaptive thresholds:
%   - I1,T1,P1 for predator S1 and I2,T2,P2 for predator S2.
%   - These threshold trajectories are the raw adaptive signals later analyzed by MDEA and DFA.
%   - Randomness: each ensemble uses an independent seed generated at run time.
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

Time   = 4e3;      % number of trials
ENS    = 1;        % number of ensembles

L      = 2 ;        % length of the 2D environment
dt     = 1;        % time step
ry     = 1 ; %  21 ;        % grid size along RS
RS     = 0.35; % 0.24 + 0.01*(1:ry)'; % 0.295 + 0.005*(1:ry)'; %  ratio of vision respect to L
deltaS = 0.1;      % learning step

% str = 0.045 ; % 0.038 ; % 0.1 ; % Stripe size for MDEA

% each agent (S1 & S2) has three adaptive propensities: information sharing (I), trust (T), and payoff sharing (P)

% ---------------- noise parameters ----------------
EnvSuccessProb = 1 ;   % probability that a geometrically available fish is actually caught

% ========================= (NO PARFOR/PARALLEL) PROGRESS SETUP =========================
% Removed: parpool/gcp, DataQueue, afterEach, send(dq,...), etc.
% Optional simple text progress:
chunkSize = max(1, floor(Time/100));  % ~100 ticks per run
% ======================================================================================

% ============================= FOR OVER RY =============================
for kk = 1:ry
    % ----- row buffers -----
    AACS_row1 = zeros(1, ENS);
    AACS_row2 = zeros(1, ENS);
    AACS_row3 = zeros(1, ENS);
    AACS_row4 = zeros(1, ENS);

    AACSPvalue_row1 = zeros(1, ENS);
    AACSPvalue_row2 = zeros(1, ENS);
    AACSPvalue_row3 = zeros(1, ENS);
    AACSPvalue_row4 = zeros(1, ENS);

    AADel_row1 = zeros(1, ENS);
    AADel_row2 = zeros(1, ENS);
    AADel_row3 = zeros(1, ENS);
    AADel_row4 = zeros(1, ENS);

    AASD_row1 = zeros(1, ENS);
    AASD_row2 = zeros(1, ENS);
    AASD_row3 = zeros(1, ENS);
    AASD_row4 = zeros(1, ENS);

    AACor_row1 = zeros(1, ENS);
    AACor_row2 = zeros(1, ENS);
    AACor_row3 = zeros(1, ENS);
    AACor_row4 = zeros(1, ENS);

    AACorPvalue_row1 = zeros(1, ENS);
    AACorPvalue_row2 = zeros(1, ENS);
    AACorPvalue_row3 = zeros(1, ENS);
    AACorPvalue_row4 = zeros(1, ENS);

    AACC_row  = zeros(1, ENS);
    AAPay_row = zeros(1, ENS);

    % -------- runs all ensembles serially for each kk --------
    for ens = 1:ENS
        % Independent RNG per (ens, kk)
        rng(randi(1e9) + kk + 1000*ens, 'twister');

        % ----------- per-kk parameters -----------
        rSfac  = RS(kk);

        % --------------- local state ---------------
        ThetaDecept = pi/6;

        % S/F params
        rS   = rSfac * L;   % radius of vision of S agents
        velS = 0.1;         % speed of S agents
        rF   = 1 * rS;      % radius of vision of F agent
        velF = 2 * velS;    % speed of F agent
        rG   = 1 * rS;      % radius of game play
        tc   = 1 ;          % temptation to cheat in the PD game

        % Initial conditions
        thetaF           = 2*pi*(rand(1, 1) - 0.5);  % angles (one F)
        thetaFoeF        = zeros(1, 1);
        thetaFoeFDECEPT  = zeros(1, 1);
        thetaS           = 2*pi*(rand(1, 2) - 0.5);  % angles (two S)
        thetaFS          = 2*pi*(rand(1, 3) - 0.5);
        thetaFoeSShared  = zeros(1, 2);

        PayF0 = zeros(1, 1);  PayF = zeros(1, 1);   % payoffs
        PayS0 = zeros(2, 1);  PayS = zeros(2, 1);

        ShareInfoS1 = ones(Time, 1);  % propensities
        ShareInfoS2 = ones(Time, 1);
        TrustS1     = ones(Time, 1);
        TrustS2     = ones(Time, 1);
        SharePayS1  = ones(Time,  1);
        SharePayS2  = ones(Time,  1);

        DecInfoS1      = zeros(Time, 1);  % decisions
        DecInfoS2      = zeros(Time, 1);
        DecTrustS1     = zeros(Time, 1);
        DecTrustS2     = zeros(Time, 1);
        DecPayShareS1  = zeros(Time, 1);
        DecPayShareS2  = zeros(Time, 1);

        PayS1 = zeros(Time, 1);  % time series of payoffs
        PayS2 = zeros(Time, 1);

        % time series of velocity x-components (for DFA etc.)
        IndividualS1Vx = zeros(Time, 1);
        IndividualS2Vx = zeros(Time, 1);
        IndividualFVx  = zeros(Time, 1);

        xF = L*rand(1, 1);  yF = L*rand(1, 1);  % positions
        xS = L*rand(1, 2);  yS = L*rand(1, 2);
        xFS = zeros(1, 3);
        yFS = zeros(1, 3);

        Ratio_CC  = zeros(Time, 1);  CC = 0;
        Ratio_Pay = zeros(Time, 1);  TotalPay = 0;

        % ----- initial velocities at ti = 1 (using initial angles) -----
        IndividualS1Vx(1) = velS * cos(thetaS(1));
        IndividualS2Vx(1) = velS * cos(thetaS(2));
        IndividualFVx(1)  = velF * cos(thetaF(1));

        % ===== MAIN TIME LOOP =====
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
            S1canshareinfo = 1;
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
                thetaS(1)      = thetaFS(2);
                S1canshareinfo = 0;
            end

            % S2 block
            S2canshareinfo = 1;
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
                thetaS(2)      = thetaFS(3);
                S2canshareinfo = 0;
            end

            % -------------- SHARE INFO --------------
            ShaInfo1       = 0;
            if rand > ShareInfoS1(ti), ShaInfo1 = 1; end
            ShaInfo2       = 0;
            if rand > ShareInfoS2(ti), ShaInfo2 = 1; end

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
                if rand < EnvSuccessProb
                    zPayS1 = 13 * LF1;
                else
                    zPayS1 = -13;
                end
            end

            if LF2 == 0
                zPayS2 = -13;
            else
                if rand < EnvSuccessProb
                    zPayS2 = 13 * LF2;
                else
                    zPayS2 = -13;
                end
            end
            % --------------------------------------------------------------------

            % -------------- SHARE PAY intentions & base payoffs --------------
            if zPayS1 == -13, PayS(1) = -2; else, PayS(1) = 2; end
            if zPayS2 == -13, PayS(2) = -2; else, PayS(2) = 2; end

            w1 = (zPayS1 == 13);
            w2 = (zPayS2 == 13);
            any_win = (w1 || w2);

            r1 = rand;  r2 = rand;
            s1 = (r1 > SharePayS1(ti));
            s2 = (r2 > SharePayS2(ti));
            SharedS1 = s1;  SharedS2 = s2;

            ShareUSedS1 = 0 ;  ShareUSedS2 = 0 ;
            if Trusted1 == 1 || Trusted2 == 1
                if any_win
                    ShareUSedS1 = 1 ;  ShareUSedS2 = 1 ;

                    % ======= PAYOFF (Prisoner's Dilemma) =======
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

            % -------------- Update thresholds --------------
            ShareInfoS1(ti + 1) = UpdateThreshold(1, ShareInfoS1(ti),  ShaInfo1,  PayS(1), PayS0(1), deltaS);
            ShareInfoS2(ti + 1) = UpdateThreshold(1, ShareInfoS2(ti),  ShaInfo2,  PayS(2), PayS0(2), deltaS);
            TrustS1(ti + 1)     = UpdateThreshold(1, TrustS1(ti),      Trusted1,  PayS(1), PayS0(1), deltaS);
            TrustS2(ti + 1)     = UpdateThreshold(1, TrustS2(ti),      Trusted2,  PayS(2), PayS0(2), deltaS);
            SharePayS1(ti + 1)  = UpdateThreshold(ShareUSedS1, SharePayS1(ti),   SharedS1,  PayS(1), PayS0(1), deltaS);
            SharePayS2(ti + 1)  = UpdateThreshold(ShareUSedS2, SharePayS2(ti),   SharedS2,  PayS(2), PayS0(2), deltaS);
            
            % -------------- Accumulate performance --------------
            PayF0 = PayF;  PayS0 = PayS;
            TotalPay      = TotalPay + mean(PayS);
            Ratio_Pay(ti) = TotalPay / ti;
            Ratio_CC(ti)  = CC / ti;

            PayS1(ti) = PayS(1);
            PayS2(ti) = PayS(2);

            % --------- simple text progress (optional) ---------
            if (mod(ti, chunkSize) == 0) || (ti == Time)
                % uncomment if you want it:
                % fprintf('kk=%d/%d, ens=%d/%d, ti=%d/%d\n', kk, ry, ens, ENS, ti, Time);
            end

            % --------- time-resolved velocity x-components (for DFA) ---------
            IndividualS1Vx(ti) = velS * cos(thetaS(1));
            IndividualS2Vx(ti) = velS * cos(thetaS(2));
            IndividualFVx(ti)  = velF * cos(thetaF(1));

            DecInfoS1(ti)     = ShaInfo1;
            DecInfoS2(ti)     = ShaInfo2;
            DecTrustS1(ti)    = Trusted1;
            DecTrustS2(ti)    = Trusted2;

        end % time

        % ================== CS + stats (single kk,ens) ==================
        % (your existing block continues here)

    end % ens

end % for kk
L = Time/2;
time = 1:L;

FS = 16;  % font size

% Teal & Orange (high contrast, non–blue/red)
c1 = [0.0 0.6 0.6 0.4];   % teal
c2 = [0.9 0.45 0.0 0.4]; % orange

figure('Position',[100 100 600 900]);  % tall figure for top-down panels

% ===== Top Panel: I(t) =====
subplot(3,1,1);
plot(time, ShareInfoS1(1:L), 'Color', c1, 'LineWidth', 2); hold on;
plot(time, ShareInfoS2(1:L), 'Color', c2, 'LineWidth', 2);
xlabel('time', 'FontSize', FS);
ylabel('I(t)', 'FontSize', FS);
set(gca, ...
    'FontSize', FS, ...
    'Box', 'off');   % frame faded, labels intact
grid off;

% ===== Middle Panel: T(t) =====
subplot(3,1,2);
plot(time, TrustS1(1:L), 'Color', c1, 'LineWidth', 2); hold on;
plot(time, TrustS2(1:L), 'Color', c2, 'LineWidth', 2);
xlabel('time', 'FontSize', FS);
ylabel('T(t)', 'FontSize', FS);
set(gca, ...
    'FontSize', FS, ...
    'Box', 'off');
grid off;

% ===== Bottom Panel: P(t) =====
subplot(3,1,3);
plot(time, SharePayS1(1:L), 'Color', c1, 'LineWidth', 2); hold on;
plot(time, SharePayS2(1:L), 'Color', c2, 'LineWidth', 2);
xlabel('time', 'FontSize', FS);
ylabel('P(t)', 'FontSize', FS);
set(gca, ...
    'FontSize', FS, ...
    'Box', 'off');
grid off;


toc






% =============================== FUNCTIONS ===============================
% Removed: localIncrement, localTextProgress (DataQueue-related)
% Keep your other functions unchanged:

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
