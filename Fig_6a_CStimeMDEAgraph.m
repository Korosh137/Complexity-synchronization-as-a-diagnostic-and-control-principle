% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 6a: local MDEA scaling-exponent time series.
%
% What this script does:
%   - Runs one simulation and computes local MDEA scaling exponents delta(t) in overlapping windows.
%   - Uses stripe-crossing events from adaptive thresholds and shows one representative MDEA scaling fit.
%   - The scaling time series are computed over all available samples; arrays are sized for ti+1 updates.
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
RRR    = 1;        % keep MDEA block
Time   = 1e6;      % number of trials
ENS    = 1;        % ONLY 1 ensemble

L      = 2;        % length of the 2D environment
dt     = 1;        % time step
deltaS = 0.1;      % learning step

Noise  = 1e-3;
RS     = 0.45;     % ONLY this RS
str    = 0.01;     % ONLY this stripe size

EnvSuccessProb = 1;  % probability that geometrically available fish is caught

%% ====================== WINDOWING FOR SCALING TIME SERIES ======================
Slice    = 1e4;
Overlap  = floor(0.75 * Slice);
Newdata  = Slice - Overlap;

% MDEA fit region (same as your MDEA defaults)
FitST = 0.1;
FitEN = 0.9;

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
% OPTION A: different randomness each script run
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

%% ====================== BUILD DATA (NOW: USE ALL DATA) ======================
% OLD (caused late start):
% ST0 = floor(0.25 * length(SharePayS1));
% NEW:
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

% Number of sliding windows (include the first window => +1)
nn = floor((N - Slice)/Newdata) + 1;
if nn <= 0
    error('Not enough data for the requested Slice/Overlap. Reduce Slice or increase Time.');
end

selCols = [1 3 6];                % I1, T1, P2
ScaleSel = zeros(nn, numel(selCols));
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
        DataX = DaTaa - min(DaTaa);
        if max(DataX) ~= 0
            DataX = DataX ./ max(DataX);
            ScaleSel(gg, jj) = MDEA((DataX), str, 1, FitST, FitEN, 0);
        else
            ScaleSel(gg, jj) = 0;
        end
    end
end

gMid = max(1, round(nn/2));
staMid = (gMid - 1) * Newdata;
i1m = 1 + staMid;
i2m = Slice + staMid;

tStartSel = (ST0 - 1) + i1m;
tEndSel   = (ST0 - 1) + i2m;
tMidSel   = tCenter(gMid);

MDEAcurves = struct();
for jj = 1:numel(selCols)
    DaTaa = data(i1m:i2m, selCols(jj));
    DataX = DaTaa - min(DaTaa);
    if max(DataX) ~= 0, DataX = DataX ./ max(DataX); end

    [deltaHat, de, DE, FitLine, Starr, endd] = MDEA_outputs(DataX, str, 1, FitST, FitEN);

    MDEAcurves(jj).delta   = deltaHat;
    MDEAcurves(jj).de      = de(:);
    MDEAcurves(jj).DE      = DE(:);
    MDEAcurves(jj).FitLine = FitLine;
    MDEAcurves(jj).Starr   = Starr;
    MDEAcurves(jj).endd    = endd;
end

%% ====================== PLOTTING: 1x2 (LEFT scaling TS, RIGHT MDEA) ======================
fig = figure('Units','pixels','Position',[120 120 1400 560]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% ---------- LEFT: scaling time series ----------
ax1 = nexttile(1); hold(ax1,'on'); box(ax1,'on'); grid(ax1,'off');
set(ax1,'FontName',fontName,'FontSize',fontSize,'LineWidth',1.2,'FontWeight','bold');

lab3 = {'I1','T1','P2'};

% ONLY lines (transparent)
for jj = 1:3
    c = cols(jj,:);
    h = plot(ax1, tCenter, ScaleSel(:,jj), '-', 'Color', c, 'LineWidth', 2.0);

    % Try true line alpha (newer MATLAB). If not supported, fall back to patch.
    try
        h.Color = [c lineAlphaLeft];  % RGBA (if supported)
    catch
        delete(h);
        patch(ax1, 'XData', tCenter(:)', 'YData', ScaleSel(:,jj)', ...
            'FaceColor','none', 'EdgeColor', c, 'LineWidth', 2.0, 'EdgeAlpha', lineAlphaLeft);
    end
end

% Mark ONLY the selected middle window points (circles, opaque)
for jj = 1:3
    c = cols(jj,:);
    plot(ax1, tMidSel, ScaleSel(gMid,jj), 'o', ...
        'MarkerSize', 10, 'LineWidth', 2.0, 'Color', c, ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', c);
end

title(ax1, sprintf('Scaling time series (MDEA \\delta) | RS=%.2f, str=%.2f', RS, str), ...
    'FontWeight','bold');
xlabel(ax1, 't (window center)', 'FontWeight','bold');
ylabel(ax1, '\delta (scaling exponent)', 'FontWeight','bold');

hP = gobjects(3,1);
for jj = 1:3
    c = cols(jj,:);
    hP(jj) = plot(ax1, nan, nan, '-', 'Color', c, 'LineWidth', 2.0);
end
lg1 = legend(ax1, hP, lab3, 'Location','northwest');
set(lg1,'FontName',fontName,'FontSize',14,'FontWeight','bold','Box','off');

hold(ax1,'off');

% ---------- RIGHT: MDEA graphs for middle window ----------
ax2 = nexttile(2); hold(ax2,'on'); box(ax2,'on'); grid(ax2,'off');
set(ax2,'FontName',fontName,'FontSize',fontSize,'LineWidth',1.2,'FontWeight','bold');

title(ax2, sprintf('MDEA in selected middle window | [%d, %d] (ti)', tStartSel, tEndSel), ...
    'FontWeight','bold');
xlabel(ax2, 'log(l)', 'FontWeight','bold');
ylabel(ax2, 'S(l)',   'FontWeight','bold');

% Plot curves + fit lines + ST/EN lines
hR = gobjects(3,1);
for jj = 1:3
    c   = cols(jj,:);
    de  = MDEAcurves(jj).de;
    DE  = MDEAcurves(jj).DE;
    Fit = MDEAcurves(jj).FitLine;
    Starr = MDEAcurves(jj).Starr;
    endd  = MDEAcurves(jj).endd;

    iA = 3;
    iB = max(4, numel(de)-3);

    % EMPTY squares for data
    hR(jj) = plot(ax2, de(iA:iB), DE(iA:iB), 's', ...
        'Color', c, 'LineWidth', 1.3, 'MarkerSize', 6, ...
        'MarkerFaceColor', 'none', 'MarkerEdgeColor', c);

    % Fit line ONLY over the fit region (Starr:endd), same color
    de0  = de(Starr:endd);
    yfit = Fit(1)*de0 + Fit(2);
    plot(ax2, de0, yfit, '--', 'Color', c, 'LineWidth', 2.0);

    % ST/EN vertical dotted lines
    xline(ax2, de(Starr), ':', 'Color', c, 'LineWidth', 1.5);
    xline(ax2, de(endd),  ':', 'Color', c, 'LineWidth', 1.5);
end

% Legend labels with slopes beside them
legLabR = cell(3,1);
for jj = 1:3
    legLabR{jj} = sprintf('%s  (\\delta=%.3f)', lab3{jj}, MDEAcurves(jj).delta);
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

function [A, B] = Finddistance_torus_unused(~,~,~,~) %#ok<DEFNU>
    A=[]; B=[];
end

function delta = MDEA(Data, Stripesize, Rule, ST, EN, PLOT)
    Data = Data - min(Data);
    if max(Data) > 0, Data = Data ./ max(Data); end

    Lengthdata = length(Data);
    Ddata      = Data./(Stripesize);
    Event      = zeros(Lengthdata, 1);

    k = 1;
    Event(1) = 1;
    StartEvent = zeros();

    for i = 2:Lengthdata
        if (Ddata(i) < floor(Ddata(i-1))) || (Ddata(i) > ceil(Ddata(i-1)))
            Event(i) = 1;
            StartEvent(k) = i;
            k = k + 1;
        end
    end

    StartEvent = StartEvent(StartEvent ~= 0);
    if Rule == 1
        Diff = cumsum(Event);
    end

    if Rule == -1
        State0 = zeros(Lengthdata, 1);
        for yy = 1:Lengthdata
            if Event(yy) == 1
                r = rand;
                if r < 0.5
                    State0(yy) = 1;
                else
                    State0(yy) = -1;
                end
            end
        end
        Diff = cumsum(State0);
    end

    if Rule == 0  % velocity
        State0 = zeros(Lengthdata, 1);
        for yy = 1:Lengthdata
            if Event(yy) == 1
                r = rand;
                if r < 0.5
                    State0(yy) = 1;
                else
                    State0(yy) = -1;
                end
            end
        end
        Diff = cumsum(State0);

        State00 = zeros(Lengthdata, 1);
        State0(1) = 1;
        for ee = 2:Lengthdata
            if State0(ee) == 0
                State00(ee) = State00(ee - 1);
            else
                State00(ee) = State0(ee);
            end
        end
        Diff = cumsum(State00);
    end

    ll  = floor(log(length(Diff))/log(1.2)) - 5;
    Delh = zeros(1, ll);

    de = zeros(1, ll);
    DE = zeros(1, ll);

    for i = 1:ll
        Delh(i) = floor(1.2^i);
    end

    for q = 1:length(Delh)
        SliceNum = 1 * length(StartEvent);
        del      = Delh(q);

        HH  = zeros(SliceNum, 1);
        enn = length(Diff);
        enn2 = length(StartEvent);
        i = 1;
        while i <= enn2 && (StartEvent(i) + del < enn)
            iiii = StartEvent(i);
            HH(i) = Diff(iiii + del) - Diff(iiii);
            i = i + 1;
        end

        XF = HH(HH ~= 0);

        nbins = floor(max(abs(XF)) / 1);
        nbins(nbins == 0) = [];
        [counts] = hist(XF, nbins);

        counts = counts(counts ~= 0);
        Pc = counts ./ sum(counts);

        DE0 = -sum((Pc) .* log(Pc) / log(10));
        DE(:, q) = DE0;
    end

    for t = 1:length(Delh)
        de(t) = log(Delh(t)) / log(10);
    end

    Starr = round(ST * length(de));
    endd  = round(EN * length(de));
    DE0   = DE(Starr:endd);
    de0   = de(Starr:endd);

    FitLine = polyfit(de0, DE0, 1);
    delta   = FitLine(1);

    if PLOT == 1
        figure
        subplot(1, 2, 1)
        plot(Ddata)
        xlabel('t','FontWeight','bold'), ylabel('X(t)','FontWeight','bold');
        legend('X(t)', 'Location', 'northwest');
        title('Signal','FontWeight','bold');

        subplot(1, 2, 2)
        plot(de(3:length(de)-3), DE(3:length(DE)-3), 's', ...
             'MarkerFaceColor','none','MarkerEdgeColor','k','LineWidth',1.0);
        hold on
        plot(de0, FitLine(1)*de0 + FitLine(2), 'r--', 'LineWidth', 1.5);
        xlabel('log(l)','FontWeight','bold'), ylabel('S(l)','FontWeight','bold');
        legend(['\delta = ' num2str(sprintf('%.3f', delta))], 'Location', 'northwest');
    end
end

function [delta, de, DE, FitLine, Starr, endd] = MDEA_outputs(Data, Stripesize, Rule, ST, EN)
    Data = Data - min(Data);
    if max(Data) > 0, Data = Data ./ max(Data); end

    Lengthdata = length(Data);
    Ddata      = Data./(Stripesize);
    Event      = zeros(Lengthdata, 1);

    k = 1;
    Event(1) = 1;
    StartEvent = zeros();

    for i = 2:Lengthdata
        if (Ddata(i) < floor(Ddata(i-1))) || (Ddata(i) > ceil(Ddata(i-1)))
            Event(i) = 1;
            StartEvent(k) = i;
            k = k + 1;
        end
    end

    StartEvent = StartEvent(StartEvent ~= 0);

    if Rule == 1
        Diff = cumsum(Event);
    elseif Rule == -1
        State0 = zeros(Lengthdata, 1);
        for yy = 1:Lengthdata
            if Event(yy) == 1
                r = rand;
                if r < 0.5, State0(yy) = 1; else, State0(yy) = -1; end
            end
        end
        Diff = cumsum(State0);
    else % Rule == 0
        State0 = zeros(Lengthdata, 1);
        for yy = 1:Lengthdata
            if Event(yy) == 1
                r = rand;
                if r < 0.5, State0(yy) = 1; else, State0(yy) = -1; end
            end
        end
        State00 = zeros(Lengthdata, 1);
        State0(1) = 1;
        for ee = 2:Lengthdata
            if State0(ee) == 0
                State00(ee) = State00(ee - 1);
            else
                State00(ee) = State0(ee);
            end
        end
        Diff = cumsum(State00);
    end

    ll  = floor(log(length(Diff))/log(1.2)) - 5;
    Delh = zeros(1, ll);
    de  = zeros(1, ll);
    DE  = zeros(1, ll);

    for i = 1:ll
        Delh(i) = floor(1.2^i);
    end

    for q = 1:length(Delh)
        SliceNum = 1 * length(StartEvent);
        del      = Delh(q);

        HH  = zeros(SliceNum, 1);
        enn = length(Diff);
        enn2 = length(StartEvent);
        i = 1;
        while i <= enn2 && (StartEvent(i) + del < enn)
            iiii = StartEvent(i);
            HH(i) = Diff(iiii + del) - Diff(iiii);
            i = i + 1;
        end

        XF = HH(HH ~= 0);

        nbins = floor(max(abs(XF)) / 1);
        nbins(nbins == 0) = [];
        [counts] = hist(XF, nbins);

        counts = counts(counts ~= 0);
        Pc = counts ./ sum(counts);

        DE0 = -sum((Pc) .* log(Pc) / log(10));
        DE(:, q) = DE0;
    end

    for t = 1:length(Delh)
        de(t) = log(Delh(t)) / log(10);
    end

    Starr = max(1, round(ST * length(de)));
    endd  = min(length(de), round(EN * length(de)));

    DE0   = DE(Starr:endd);
    de0   = de(Starr:endd);

    FitLine = polyfit(de0, DE0, 1);
    delta   = FitLine(1);
end
