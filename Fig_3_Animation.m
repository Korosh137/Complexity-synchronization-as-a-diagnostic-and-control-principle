% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 3 animation: Predator-Prey model dynamics.
%
% What this script does:
%   - Runs one stochastic simulation of the two-predator/one-prey adaptive model.
%   - Creates an AVI animation showing agent positions, sensing radii, and headings.
%   - Plots the cooperation-rate time series after the simulation.
%   - No ensemble averaging, parameter sweep, MDEA, DFA, or correlation analysis is used here.
%   - Randomness: uses MATLAB default random stream unless the user sets rng before running.
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

RRR    = 1;
Time   = 1e5;      % number of trials

L      = 2;        % length of the 2D environment
dt     = 1;        % time step
deltaS = 0.1;      % learning step

RS    = 0.35;
Noise = 1e-3;

ThetaDecept = pi/6;

% ---------------- video / animation settings ----------------
makeVideo   = true;
videoName   = 'SF_main_animation.avi';
fps         = 30;          % playback frame rate
startFrac   = 0.90;        % start recording when ti/Time > startFrac (like your example)
frameSkip   = 1;           % write every frame; set 2/5/10 to reduce file size
arrowLen    = 0.12 * L;    % arrow length for quiver

if makeVideo
    mov = VideoWriter(videoName);
    mov.FrameRate = fps;
    open(mov);
end

% ---------------- noise parameters ----------------
EnvSuccessProb = 1;   % (your main code version: deterministic catch currently; kept for compatibility)

% ======================== S/F params ========================
rS   = RS * L;
velS = 0.1;
rF   = 1 * rS;
velF = 2 * velS;
rG   = 1 * rS;
tc   = 3;

% ======================== Initial conditions ========================
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

Ratio_CC  = zeros(Time, 1);
CC = 0;

% If you want a window title that updates:
figAnim = figure('Units','pixels','Position',[80 80 650 620]);
set(figAnim,'Color','w');

% ===================== MAIN TIME LOOP =====================
for ti = 2:Time

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

    % -------- decisions --------
    ShaInfo1 = 0; if rand > ShareInfoS1(ti), ShaInfo1 = 1; end
    ShaInfo2 = 0; if rand > ShareInfoS2(ti), ShaInfo2 = 1; end

    Trusted1  = 0; if rand > TrustS1(ti), Trusted1 = 1; thetaS(1) = thetaFoeSShared(2); end
    Trusted2  = 0; if rand > TrustS2(ti), Trusted2 = 1; thetaS(2) = thetaFoeSShared(1); end

    % Integrate + wrap (periodic torus)
    xF = mod(xF + velF * cos(thetaF) * dt , L);
    yF = mod(yF + velF * sin(thetaF) * dt , L);
    xS = mod(xS + velS * cos(thetaS) * dt , L);
    yS = mod(yS + velS * sin(thetaS) * dt , L);

    % payoff neighborhood
    xFS = [xF(1), xS(1, 1), xS(1, 2)];
    yFS = [yF(1), yS(1, 1), yS(1, 2)];
    [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rG, L); %#ok<ASGLU>

    list = l1FS(l2FS == 2);  listofF = list(list <= 1);  LF1 = length(listofF);
    list = l1FS(l2FS == 3);  listofF = list(list <= 1);  LF2 = length(listofF);

    % NOTE: your main code version is deterministic here (no EnvSuccessProb)
    if LF1 == 0
        zPayS1 = -13;
    else
        zPayS1 = 13 * LF1;
    end

    if LF2 == 0
        zPayS2 = -13;
    else
        zPayS2 = 13 * LF2;
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

    ShareInfoS1(ti + 1) = UpdateThreshold(1, ShareInfoS1(ti),  ShaInfo1,  PayS(1), PayS0(1), deltaS, Noise);
    ShareInfoS2(ti + 1) = UpdateThreshold(1, ShareInfoS2(ti),  ShaInfo2,  PayS(2), PayS0(2), deltaS, Noise);
    TrustS1(ti + 1)     = UpdateThreshold(1, TrustS1(ti),      Trusted1,  PayS(1), PayS0(1), deltaS, Noise);
    TrustS2(ti + 1)     = UpdateThreshold(1, TrustS2(ti),      Trusted2,  PayS(2), PayS0(2), deltaS, Noise);
    SharePayS1(ti + 1)  = UpdateThreshold(ShareUSedS1, SharePayS1(ti), s1, PayS(1), PayS0(1), 1*deltaS, Noise);
    SharePayS2(ti + 1)  = UpdateThreshold(ShareUSedS2, SharePayS2(ti), s2, PayS(2), PayS0(2), deltaS, Noise);

    PayS0 = PayS;

    Ratio_CC(ti) = CC / ti;

    % ===================== ANIMATION / VIDEO (LIKE YOUR EXAMPLE) =====================
    doRecord = makeVideo && (ti/Time > startFrac) && (mod(ti, frameSkip) == 0);
    if doRecord
        figure(figAnim); clf;
        hold on; box on;

        % vision circles (periodic copies)
        drawCirclePeriodic(xF(1), yF(1), rF, L, 'r');  % F vision
        drawCirclePeriodic(xS(1), yS(1), rS, L, 'b');  % S1 vision
        drawCirclePeriodic(xS(2), yS(2), rS, L, 'b');  % S2 vision

        % agents
        plot(xF, yF, 'r.', 'MarkerSize', 22);          % F as red dot
        plot(xS, yS, 'bv', 'MarkerSize', 8);           % S as blue triangles

        % direction arrows
        quiver(xF(1), yF(1), arrowLen*cos(thetaF(1)), arrowLen*sin(thetaF(1)), ...
               0, 'Color','r', 'LineWidth',1.2, 'MaxHeadSize',1.5);
        quiver(xS(1), yS(1), arrowLen*cos(thetaS(1)), arrowLen*sin(thetaS(1)), ...
               0, 'Color','b', 'LineWidth',1.2, 'MaxHeadSize',1.5);
        quiver(xS(2), yS(2), arrowLen*cos(thetaS(2)), arrowLen*sin(thetaS(2)), ...
               0, 'Color','b', 'LineWidth',1.2, 'MaxHeadSize',1.5);

        % frame, labels, title
        xlim([0 L]); ylim([0 L]); axis square;
        title(sprintf('ti = %d / %d   |   CC = %.4f   |   RS = %.2f   |   Noise = %.1e', ...
              ti, Time, Ratio_CC(ti), RS, Noise), 'FontWeight','bold');

        drawnow;

        frame = getframe(figAnim);
        writeVideo(mov, frame);
    end
end % time loop

if makeVideo
    close(mov);
end

% quick diagnostic plot (not cross-correlation; only CC ratio over time)
figure; plot(Ratio_CC, 'LineWidth', 1.6);
xlabel('time'); ylabel('Mutual Cooperation (CC ratio)');
title('CC Ratio vs Time');

toc

% =============================== FUNCTIONS ===============================
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

function aa = UpdateThreshold(Used, pi0, Decision0, Pay, Paybefore, ChangeThreshold, noiseInt)
    if Used == 1
        if Decision0 == 1, dp = -1*ChangeThreshold; else, dp = 1*ChangeThreshold; end
        if Pay ~= 0 && Paybefore ~= 0
            DeltaPay = (Pay - Paybefore);
        else
            DeltaPay = 0;
        end
        DeltaPay = DeltaPay + noiseInt*randn;
        aa = pi0 + dp * DeltaPay;
    else
        aa = pi0 +  noiseInt*randn;
    end
    aa = min(max(aa, 0), 1);
end

function drawCirclePeriodic(cx, cy, r, L, colorSpec)
% Draw a circle of radius r centered at (cx,cy) on a periodic LxL domain
% by plotting 9 wrapped copies (center and neighbors).
    th = linspace(0, 2*pi, 128);
    shifts = [-L 0 L];
    for dx = shifts
        for dy = shifts
            x = cx + dx + r*cos(th);
            y = cy + dy + r*sin(th);
            plot(x, y, colorSpec, 'LineWidth', 0.7);
        end
    end
end
