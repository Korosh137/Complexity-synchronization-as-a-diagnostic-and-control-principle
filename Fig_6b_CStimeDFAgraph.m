% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 6b: local DFA scaling-exponent time series.
%
% What this script does:
%   - Runs one simulation and computes local DFA scaling exponents H(t) in overlapping windows.
%   - Shows DFA scaling curves and a representative fit over W = 100 to 1000.
%   - No stripe-size parameter is used because DFA analyzes the original threshold signals directly.
%   - Randomness: rng('shuffle','twister') is used, so repeated runs are independent.
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

%% ====================== FIXED SETTINGS (per request) ======================
Time   = 1e6;      % number of trials
ENS    = 1;        % ONLY 1 ensemble

L      = 2;        % length of the 2D environment
dt     = 1;        % time step
deltaS = 0.1;      % learning step

Noise  = 1e-3;
RS     = 0.25;     % ONLY this RS

EnvSuccessProb = 1;  % probability that geometrically available fish is caught

%% ====================== WINDOWING FOR SCALING TIME SERIES ======================
Slice    = 1e4;
Overlap  = floor(0.75 * Slice);
Newdata  = Slice - Overlap;

%% ====================== DFA SETTINGS ======================
dfa_order = 1;
dfa_plot  = 0;

% Time-series DFA (left panel): keep as before (fit over 100..1000)
dfa_pts_ts   = 100:100:1000;

% Right-panel DFA curve: 10..5000 + add two points between 10..100
dfa_pts_plot = unique([10 20 50 100:100:1000 1500:500:5000]);  % includes 10..5000

% Fitting range for right panel
FitW_ST = 100;
FitW_EN = 1000;

%% ====================== COLORS (keep EXACT palette style) ======================
cols = [
    0.00    0.45    0.90
    0.90    0.30    0.05
    0.00    0.60    0.60
];

fontName = 'Arial';
fontSize = 12;

% transparency for LEFT-panel lines
lineAlphaLeft = 0.6;

%% ====================== RUN SINGLE SIMULATION (ENS=1) ======================
rng('shuffle','twister');

ThetaDecept = pi/6;

% S/F params
rS   = RS * L;     % radius of vision of S agents
velS = 0.1;        % speed of S agents
rF   = 1 * rS;     % radius of vision of F agent
velF = 2 * velS;   % speed of F agent
rG   = 1 * rS;     % radius of game play
tc   = 3;          % temptation to cheat in the PD game

% Initial conditions
thetaF           = 2*pi*(rand(1, 1) - 0.5);
thetaFoeF        = zeros(1, 1);
thetaFoeFDECEPT  = zeros(1, 1);
thetaS           = 2*pi*(rand(1, 2) - 0.5);
thetaFS          = 2*pi*(rand(1, 3) - 0.5);
thetaFoeSShared  = zeros(1, 2);

PayS0 = zeros(2, 1);  PayS = zeros(2, 1);

% IMPORTANT: allocate Time+1 because we write (ti+1)
ShareInfoS1 = ones(Time+1, 1);
ShareInfoS2 = ones(Time+1, 1);
TrustS1     = ones(Time+1, 1);
TrustS2     = ones(Time+1, 1);
SharePayS1  = ones(Time+1, 1);
SharePayS2  = ones(Time+1, 1);

xF = L*rand(1, 1);  yF = L*rand(1, 1);
xS = L*rand(1, 2);  yS = L*rand(1, 2);
xFS = zeros(1, 3);
yFS = zeros(1, 3);

Ratio_CC  = zeros(Time, 1);  CC = 0;
Ratio_Pay = zeros(Time, 1);  TotalPay = 0;

% Optional single progress bar (time loop)
chunkSize = max(1, floor(Time/200));
hWait = [];
try
    hWait = waitbar(0,'Running (0%)','Name','Simulation Progress');
catch
    hWait = [];
end

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

    ShareInfoS1(ti + 1) = UpdateThreshold(1, ShareInfoS1(ti),  ShaInfo1,  PayS(1), PayS0(1), deltaS, Noise);
    ShareInfoS2(ti + 1) = UpdateThreshold(1, ShareInfoS2(ti),  ShaInfo2,  PayS(2), PayS0(2), deltaS, Noise);
    TrustS1(ti + 1)     = UpdateThreshold(1, TrustS1(ti),      Trusted1,  PayS(1), PayS0(1), deltaS, Noise);
    TrustS2(ti + 1)     = UpdateThreshold(1, TrustS2(ti),      Trusted2,  PayS(2), PayS0(2), deltaS, Noise);
    SharePayS1(ti + 1)  = UpdateThreshold(ShareUSedS1, SharePayS1(ti),   s1,       PayS(1), PayS0(1), 1*deltaS, Noise);
    SharePayS2(ti + 1)  = UpdateThreshold(ShareUSedS2, SharePayS2(ti),   s2,       PayS(2), PayS0(2), deltaS, Noise);

    PayS0 = PayS;
    TotalPay      = TotalPay + mean(PayS);
    Ratio_Pay(ti) = TotalPay / ti;
    Ratio_CC(ti)  = CC / ti;

    if ~isempty(hWait) && (mod(ti, chunkSize) == 0 || ti == Time)
        try
            waitbar(ti/Time, hWait, sprintf('Running (%.1f%%%%)', 100*ti/Time));
        catch
        end
    end
end

try
    if ~isempty(hWait) && isvalid(hWait), close(hWait); end
catch
end

%% ====================== BUILD DATA (USE ALL DATA) ======================
ST0 = 1;
EN0 = length(SharePayS1) - 1;

data = zeros(EN0 - ST0 + 1, 6);
data(:, 1) = ShareInfoS1( ST0:EN0);   % I1
data(:, 2) = ShareInfoS2( ST0:EN0);   % I2
data(:, 3) = TrustS1(     ST0:EN0);   % T1
data(:, 4) = TrustS2(     ST0:EN0);   % T2
data(:, 5) = SharePayS1(  ST0:EN0);   % P1
data(:, 6) = SharePayS2(  ST0:EN0);   % P2

N = size(data,1);
if N < Slice
    error('Not enough data for the requested Slice/Overlap. Reduce Slice or increase Time.');
end

nn = floor((N - Slice)/Newdata) + 1;
if nn <= 0
    error('Not enough data for the requested Slice/Overlap. Reduce Slice or increase Time.');
end

%% ====================== DFA SCALING TIME SERIES (I1, T1, P2) ======================
selCols  = [1 3 6];                 % I1, T1, P2
ScaleSel = nan(nn, numel(selCols)); % alpha(t)
tCenter  = zeros(nn, 1);

for gg = 1:nn
    sta = (gg - 1) * Newdata;
    i1  = 1 + sta;
    i2  = Slice + sta;

    t_start_orig = (ST0 - 1) + i1;
    t_end_orig   = (ST0 - 1) + i2;
    tCenter(gg)  = round(0.5 * (t_start_orig + t_end_orig));

    for jj = 1:numel(selCols)
        DaTaa = data(i1:i2, selCols(jj));

        DaTaa = DaTaa(:);
        DaTaa = DaTaa(isfinite(DaTaa));

        if ~isempty(DaTaa) && (max(DaTaa) ~= 0)
            A = DFA_func(DaTaa, dfa_pts_ts, dfa_order, dfa_plot);
            ScaleSel(gg, jj) = A(1);   % alpha
        else
            ScaleSel(gg, jj) = NaN;
        end
    end
end

%% ====================== SELECT MIDDLE WINDOW FOR RIGHT PANEL ======================
gMid   = max(1, round(nn/2));
staMid = (gMid - 1) * Newdata;
i1m    = 1 + staMid;
i2m    = Slice + staMid;

tStartSel = (ST0 - 1) + i1m;
tEndSel   = (ST0 - 1) + i2m;
tMidSel   = tCenter(gMid);

%% ====================== DFA CURVES IN MIDDLE WINDOW (RIGHT PANEL) ======================
DFAcurves = struct();
for jj = 1:numel(selCols)
    DaTaa = data(i1m:i2m, selCols(jj));
    DaTaa = DaTaa(:);
    DaTaa = DaTaa(isfinite(DaTaa));

    % Compute full curve 10..5000 (incl. extra 20,50)
    [Aall, Fall] = DFA_func_full(DaTaa, dfa_pts_plot, dfa_order);

    % Fit ONLY over 100..1000
    [AfitRange] = DFA_fit_range(dfa_pts_plot, Fall, FitW_ST, FitW_EN);

    DFAcurves(jj).alpha = AfitRange(1);
    DFAcurves(jj).Afit  = AfitRange;      % [slope intercept] on log10 scale
    DFAcurves(jj).pts   = dfa_pts_plot(:);
    DFAcurves(jj).F     = Fall(:);
end

%% ====================== PLOTTING: 1x2 (LEFT alpha TS, RIGHT DFA log-log) ======================
fig = figure('Units','pixels','Position',[120 120 1400 560]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

lab3 = {'I1','T1','P2'};

% ---------- LEFT: DFA alpha time series ----------
ax1 = nexttile(1); hold(ax1,'on'); box(ax1,'on'); grid(ax1,'off');
set(ax1,'FontName',fontName,'FontSize',fontSize,'LineWidth',1.2,'FontWeight','bold');

for jj = 1:3
    c = cols(jj,:);
    h = plot(ax1, tCenter, ScaleSel(:,jj), '-', 'Color', c, 'LineWidth', 2.0);
    try
        h.Color = [c lineAlphaLeft];
    catch
        delete(h);
        patch(ax1, 'XData', tCenter(:)', 'YData', ScaleSel(:,jj)', ...
            'FaceColor','none', 'EdgeColor', c, 'LineWidth', 2.0, 'EdgeAlpha', lineAlphaLeft);
    end
end

for jj = 1:3
    c = cols(jj,:);
    plot(ax1, tMidSel, ScaleSel(gMid,jj), 'o', ...
        'MarkerSize', 10, 'LineWidth', 2.0, 'Color', c, ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', c);
end

title(ax1, sprintf('Scaling time series (DFA \\alpha) | RS=%.2f', RS), 'FontWeight','bold');
xlabel(ax1, 't (window center)', 'FontWeight','bold');
ylabel(ax1, '\alpha (DFA exponent)', 'FontWeight','bold');

hP = gobjects(3,1);
for jj = 1:3
    c = cols(jj,:);
    hP(jj) = plot(ax1, nan, nan, '-', 'Color', c, 'LineWidth', 2.0);
end
lg1 = legend(ax1, hP, lab3, 'Location','northwest');
set(lg1,'FontName',fontName,'FontSize',14,'FontWeight','bold','Box','off');
hold(ax1,'off');

% ---------- RIGHT: DFA log-log in selected middle window ----------
ax2 = nexttile(2); hold(ax2,'on'); box(ax2,'on'); grid(ax2,'off');
set(ax2,'FontName',fontName,'FontSize',fontSize,'LineWidth',1.2,'FontWeight','bold');

title(ax2, sprintf('DFA in selected middle window | [%d, %d] (ti)', tStartSel, tEndSel), ...
    'FontWeight','bold');
xlabel(ax2, 'log_{10}(W)',     'FontWeight','bold');
ylabel(ax2, 'log_{10}(F(W))',  'FontWeight','bold');

% Vertical lines at W=100 and W=1000 (fit range)
xline(ax2, log10(FitW_ST), ':', 'Color', [0 0 0], 'LineWidth', 1.4);
xline(ax2, log10(FitW_EN), ':', 'Color', [0 0 0], 'LineWidth', 1.4);

hR = gobjects(3,1);
for jj = 1:3
    c   = cols(jj,:);
    pts = DFAcurves(jj).pts;
    F   = DFAcurves(jj).F;

    x = log10(pts);
    y = log10(F);

    ok = isfinite(x) & isfinite(y) & (F > 0);
    x = x(ok); y = y(ok);

    % EMPTY squares
    hR(jj) = plot(ax2, x, y, 's', ...
        'Color', c, 'LineWidth', 1.3, 'MarkerSize', 6, ...
        'MarkerFaceColor', 'none', 'MarkerEdgeColor', c);

    % Fit line ONLY in the fitting range [100,1000]
    Afit = DFAcurves(jj).Afit;
    xfit = linspace(log10(FitW_ST), log10(FitW_EN), 200);
    yfit = polyval(Afit, xfit);
    plot(ax2, xfit, yfit, '--', 'Color', c, 'LineWidth', 2.0);
end

legLabR = cell(3,1);
for jj = 1:3
    legLabR{jj} = sprintf('%s  (\\alpha=%.3f)', lab3{jj}, DFAcurves(jj).alpha);
end
lg2 = legend(ax2, hR, legLabR, 'Location','northwest');
set(lg2,'FontName',fontName,'FontSize',14,'FontWeight','bold','Box','off', 'Interpreter','tex');

hold(ax2,'off');

toc

%% =============================== FUNCTIONS ===============================
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
        aa = pi0 + noiseInt*randn;
    end
    aa = min(max(aa, 0), 1);
end

%%%  https://www.mathworks.com/matlabcentral/fileexchange/67889-detrended-fluctuation-analysis-dfa
function [A,F] = DFA_func(data, pts, order, PLOT)
if nargin < 4, PLOT = 0; end
if nargin < 3 || isempty(order), order = 1; end

sz = size(data);
if sz(1) < sz(2), data = data'; end

if min(pts) < (order+1)
    disp(['ERROR: Smallest window size is ' num2str(min(pts)) ', DFA order is ' num2str(order) '.'])
    disp(['Aborting. Smallest window size should be at least ' num2str(order+1) '.'])
    A = [NaN NaN];
    F = NaN(size(pts(:)));
    return
end

npts = numel(pts);
F = zeros(npts,1);
N = length(data);

for h = 1:npts
    w = pts(h);

    n = floor(N/w);
    Nfloor = n*w;
    D = data(1:Nfloor);

    y  = cumsum(D-mean(D));

    bin = 0:w:(Nfloor-1);
    vec = 1:w;

    coeff = arrayfun(@(j) polyfit(vec', y(bin(j) + vec), order), 1:n, 'uni', 0);
    y_hat = cell2mat(cellfun(@(c) polyval(c, vec), coeff, 'uni', 0));
    F(h)  = mean((y - y_hat').^2)^0.5;
end

A = polyfit(log10(pts), log10(F)', 1);

if PLOT == 1
    figure;
    scatter(log10(pts), log10(F))
    hold on
    x = pts;
    plot(log10(x), polyval(A, log10(x)), '--')
    xlabel('log_{10} W'), ylabel('log_{10} F(W)');
    legend(['\alpha = ' num2str(sprintf('%.3f', A(1)))], 'Location', 'northwest');
    hold off
end
end

function [A,F] = DFA_func_full(data, pts, order)
    if nargin < 3 || isempty(order), order = 1; end
    if isempty(data) || ~any(isfinite(data))
        A = [NaN NaN];
        F = NaN(numel(pts),1);
        return
    end
    data = data(:);
    data = data(isfinite(data));
    if isempty(data)
        A = [NaN NaN];
        F = NaN(numel(pts),1);
        return
    end
    [A,F] = DFA_func(data, pts, order, 0);
end

% Fit only within [Wst, Wen] using the already-computed F(W)
function Afit = DFA_fit_range(pts, F, Wst, Wen)
    pts = pts(:);
    F   = F(:);

    ok = isfinite(pts) & isfinite(F) & (F > 0) & (pts >= Wst) & (pts <= Wen);
    if nnz(ok) < 2
        Afit = [NaN NaN];
        return
    end

    x = log10(pts(ok));
    y = log10(F(ok));
    Afit = polyfit(x, y, 1);
end
