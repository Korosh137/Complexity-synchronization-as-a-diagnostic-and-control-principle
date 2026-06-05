% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 7: cooperation rate versus MDEA-based complexity synchronization.
%
% What this script does:
%   - Sweeps sensing-radius ratio R and computes MDEA-based CS for three representative threshold pairs.
%   - Rows correspond to stripe sizes 0.1, 0.01, and 0.001; columns correspond to I1-I2, I2-T1, and I1-P2.
%   - CS is the Pearson correlation between local MDEA scaling-exponent time series.
%   - Filled/open markers indicate significant/non-significant CS links based on p < 0.05.
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
tic %%%
clc
clear all
close all

RRR    = 1;        % 1 for MDEA block to run
Time   = 1e6;      % number of trials
ENS    = 10;        % number of ensembles

L      = 2 ;       % length of the 2D environment
dt     = 1;        % time step
ry     = 21 ;      % grid size along RS
RS     =  0.24 + 0.01*(1:ry)'; % ratio of vision respect to L
deltaS = 0.1;      % learning step

Noise = 1e-3 ;

% ---------------- noise parameters ----------------
EnvSuccessProb = 1 ;   % probability that a geometrically available fish is actually caught

% ---------------- STRIPE values (AS REQUESTED) ----------------
strList = [0.1, 0.01, 0.001];
nStr    = numel(strList);

% ---------------- outputs (ONLY CS; NO velocity CS) ----------------
AACS1 = zeros(ry, ENS, nStr);  % CS between I1 & I2
AACS2 = zeros(ry, ENS, nStr);  % CS between I2 & T1
AACS3 = zeros(ry, ENS, nStr);  % CS between I1 & P2

% P-values for CS
AACSPvalue1 = nan(ry, ENS, nStr);
AACSPvalue2 = nan(ry, ENS, nStr);
AACSPvalue3 = nan(ry, ENS, nStr);

AACC  = zeros(ry, ENS);   % mutual cooperation rate
AAPay = zeros(ry, ENS);   % payoff ratio

% ========================= PARALLEL + PROGRESS SETUP =========================
if isempty(gcp('nocreate')), parpool; end

seeds = randi(1e9, ENS, 1);

chunkSize   = max(1, floor(Time/100));
ticksPerRun = ceil(Time / chunkSize);
totalTicks  = ry * ENS * ticksPerRun;

dq   = parallel.pool.DataQueue;
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

    AACS_row1 = zeros(nStr, ENS);
    AACS_row2 = zeros(nStr, ENS);
    AACS_row3 = zeros(nStr, ENS);

    AACSPvalue_row1 = ones(nStr, ENS);
    AACSPvalue_row2 = ones(nStr, ENS);
    AACSPvalue_row3 = ones(nStr, ENS);

    AACC_row  = zeros(1, ENS);
    AAPay_row = zeros(1, ENS);

    for ens = 1:ENS
        idxRS  = kk;
        idxStr = 1;
        rng(seeds(ens) + 100000*idxRS + idxStr, 'twister');

        rSfac  = RS(kk);
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

        PayF0 = zeros(1, 1);  PayF = zeros(1, 1);   %#ok<NASGU>
        PayS0 = zeros(2, 1);  PayS = zeros(2, 1);

        ShareInfoS1 = ones(Time, 1);
        ShareInfoS2 = ones(Time, 1);
        TrustS1     = ones(Time, 1);
        TrustS2     = ones(Time, 1);
        SharePayS1  = ones(Time, 1);
        SharePayS2  = ones(Time, 1);

        DecInfoS1      = zeros(Time, 1);
        DecInfoS2      = zeros(Time, 1);
        DecTrustS1     = zeros(Time, 1);
        DecTrustS2     = zeros(Time, 1);

        PayS1 = zeros(Time, 1);
        PayS2 = zeros(Time, 1);

        xF = L*rand(1, 1);  yF = L*rand(1, 1);
        xS = L*rand(1, 2);  yS = L*rand(1, 2);
        xFS = zeros(1, 3);
        yFS = zeros(1, 3);

        Ratio_CC  = zeros(Time, 1);  CC = 0;
        Ratio_Pay = zeros(Time, 1);  TotalPay = 0;

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
            if rand < 0.5
                thetaF(1) = thetaFoeF(1);
            else
                thetaF(1) = thetaFoeFDECEPT(1);
            end

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

            ShaInfo1 = 0; if rand > ShareInfoS1(ti), ShaInfo1 = 1; end
            ShaInfo2 = 0; if rand > ShareInfoS2(ti), ShaInfo2 = 1; end

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

            ShareUSedS1 = 0 ;  ShareUSedS2 = 0 ;
            if Trusted1 == 1 || Trusted2 == 1
                if any_win
                    ShareUSedS1 = 1 ;  ShareUSedS2 = 1 ;

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

            PayS1(ti) = PayS(1);
            PayS2(ti) = PayS(2);

            if (mod(ti, chunkSize) == 0) || (ti == Time)
                send(dq, 1);
            end

            DecInfoS1(ti)  = ShaInfo1;
            DecInfoS2(ti)  = ShaInfo2;
            DecTrustS1(ti) = Trusted1;
            DecTrustS2(ti) = Trusted2;
        end % time

        if 1 == RRR
            ST0 = floor(0.25 * length(SharePayS1));
            EN0 = length(SharePayS1) - 1;

            data = zeros(EN0 - ST0 + 1, 6);
            data(:, 1) = ShareInfoS1( ST0:EN0);
            data(:, 2) = ShareInfoS2( ST0:EN0);
            data(:, 3) = TrustS1(     ST0:EN0);
            data(:, 4) = TrustS2(     ST0:EN0);
            data(:, 5) = SharePayS1(  ST0:EN0);
            data(:, 6) = SharePayS2(  ST0:EN0);

            Slice    = 1e4;
            Overlap  = floor(0.75 * Slice);
            Newdata  = Slice - Overlap;
            nn       = max(0, floor((size(data, 1) - Slice) / Newdata));

            for sIdx = 1:nStr
                str = strList(sIdx);

                local_CORRYCS  = zeros(6, 6);
                local_CORRYCSp = ones(6, 6);

                Scale = zeros(max(nn, 1), 6);
                if nn > 0
                    for hh11 = 1:6
                        for gg11 = 1:nn
                            sta   = (gg11 - 1) * Newdata;
                            DaTaa = data(1 + sta : Slice + sta, hh11);

                            DataX = DaTaa - min(DaTaa);
                            if max(DataX) ~= 0
                                DataX = DataX ./ max(DataX);
                                Scale(gg11, hh11) = MDEA((DataX), str, 1, 0.1, 0.9, 0);
                            else
                                Scale(gg11, hh11) = 0;
                            end
                        end
                    end
                end

                for ii = 1:6
                    for jj = 1:6
                        if nn > 1
                            [a, p] = corrcoef(Scale(:, ii), Scale(:, jj));
                            local_CORRYCS(ii, jj)  = a(2, 1);
                            local_CORRYCSp(ii, jj) = p(2, 1);
                        else
                            local_CORRYCS(ii, jj)  = 0;
                            local_CORRYCSp(ii, jj) = 1;
                        end
                    end
                end

                AACS_row1(sIdx, ens) = local_CORRYCS(1, 2);
                AACS_row2(sIdx, ens) = local_CORRYCS(2, 3);
                AACS_row3(sIdx, ens) = local_CORRYCS(1, 6);

                AACSPvalue_row1(sIdx, ens) = local_CORRYCSp(1, 2);
                AACSPvalue_row2(sIdx, ens) = local_CORRYCSp(2, 3);
                AACSPvalue_row3(sIdx, ens) = local_CORRYCSp(1, 6);
            end
        end

        AACC_row(ens)  = Ratio_CC(end);
        AAPay_row(ens) = Ratio_Pay(end);
    end % ens

    for sIdx = 1:nStr
        AACS1(kk, :, sIdx) = AACS_row1(sIdx, :);
        AACS2(kk, :, sIdx) = AACS_row2(sIdx, :);
        AACS3(kk, :, sIdx) = AACS_row3(sIdx, :);

        AACSPvalue1(kk, :, sIdx) = AACSPvalue_row1(sIdx, :);
        AACSPvalue2(kk, :, sIdx) = AACSPvalue_row2(sIdx, :);
        AACSPvalue3(kk, :, sIdx) = AACSPvalue_row3(sIdx, :);
    end

    AACC(kk, :)  = AACC_row;
    AAPay(kk, :) = AAPay_row;
end % parfor

try
    if ~isempty(hWait) && isvalid(hWait)
        waitbar(1, hWait, 'Done (100%)');  pause(0.05);
        close(hWait);
    end
catch
end

%% =============================== PLOTTING: CC vs CS (3x3) ===============================
%% =============================== PLOTTING: CC vs CS (3x3) ===============================

% -------- FIX: CC vector must be ry x 1 for any ENS --------
if ENS == 1
    CCmean = AACC(:,1);
else
    CCmean = mean(AACC, 2, 'omitnan');   % ry x 1
end

names_cols = {'I1__I2', 'I2__T1', 'I1__P2'};

fontName      = 'Arial';
fontSize      = 12;
labelFontSize = 18;   % matched to DFA figure: larger x/y-axis labels
R2fitFontSize = 14;   % matched to DFA figure: larger R^2_{fit} text

msz_sq        = 30;
lw_hollow     = 2.0;
lw_filled     = 1.2;

bandAlpha     = 0.42;
boundLW       = 1.05;
boundAlpha    = 0.35;
lineAlphaBand = 0.55;

toCol = @(v) v(:);

xlims = [-0.1 1];
p_thr = 0.05;

yPerf  = CCmean(:);                 % ry x 1
maskY  = isfinite(yPerf);
ymin   = min(yPerf(maskY));
ymax   = max(yPerf(maskY)) + 0.2;
yr     = max(eps, ymax - ymin);
buffer = 0.05 * yr;
ylims_shared = [ymin - buffer, ymax + buffer];

fig = figure('Units','pixels','Position',[80 60 1200 900]);
set(fig,'Renderer','opengl');

tl = tiledlayout(nStr, 3);
tl.TileSpacing = 'none';
tl.Padding     = 'none';
tl.Position    = [0.05 0.06 0.86 0.90];

% grayscale gradient for RS encoding (white->black)
Ncol  = 256;
gmin  = 0.10;
gmax  = 0.97;
gamma = 0.35;
s = linspace(0,1,Ncol)';
g = gmin + (gmax-gmin) * (s.^gamma);
g = flipud(g);
colormap(fig, [g g g]);

axs = gobjects(nStr,3);

for rIdx = 1:nStr
    for cIdx = 1:3

        ax = nexttile; axs(rIdx,cIdx) = ax;
        hold(ax,'on'); box(ax,'on');

        set(ax,'FontName',fontName,'FontSize',fontSize, ...
            'Units','normalized', ...
            'PositionConstraint','outerposition', ...
            'ActivePositionProperty','outerposition', ...
            'LooseInset',[0 0 0 0]);

        switch cIdx
            case 1
                CS = AACS1(:,:,rIdx);  PV = AACSPvalue1(:,:,rIdx);
            case 2
                CS = AACS2(:,:,rIdx);  PV = AACSPvalue2(:,:,rIdx);
            case 3
                CS = AACS3(:,:,rIdx);  PV = AACSPvalue3(:,:,rIdx);
        end

        avgCS = mean(CS, 2, 'omitnan');
        sdCS  = std (CS, 0, 2, 'omitnan');
        avgP  = mean(PV,  2, 'omitnan');

        base_mask = isfinite(avgCS) & isfinite(yPerf) & isfinite(sdCS) & isfinite(avgP);
        sig_mask  = base_mask & (avgP < p_thr);
        nsig_mask = base_mask & ~sig_mask;

        x_sig  = avgCS(sig_mask);   y_sig  = yPerf(sig_mask);
        x_nsig = avgCS(nsig_mask);  y_nsig = yPerf(nsig_mask);

        idx_sig   = find(sig_mask);
        idx_nsig  = find(nsig_mask);

        c_sig  = (idx_sig  - 1) / max(ry - 1, 1);
        c_nsig = (idx_nsig - 1) / max(ry - 1, 1);

        % ===== band (horizontal SD) =====
        idx_base = find(base_mask);
        if numel(idx_base) >= 2
            idx_base = sort(idx_base,'ascend');

            yb = toCol(yPerf(idx_base));
            xb = toCol(avgCS(idx_base));
            sb = toCol(sdCS(idx_base));

            xU = xb + sb;
            xL = xb - sb;

            xU = min(xlims(2), max(xlims(1), xU));
            xL = min(xlims(2), max(xlims(1), xL));

            bandCol = [0.75 0.90 1.00];
            fill(ax, [xU; flipud(xL)], [yb; flipud(yb)], bandCol, ...
                'FaceAlpha', bandAlpha, 'EdgeColor', 'none');

            b1 = plot(ax, xU, yb, '-', 'LineWidth', boundLW, 'Color', [0 0 0]);
            b2 = plot(ax, xL, yb, '-', 'LineWidth', boundLW, 'Color', [0 0 0]);
            try
                b1.Color(4) = boundAlpha;
                b2.Color(4) = boundAlpha;
            catch
            end
        end

        % points
        if ~isempty(x_sig)
            scatter(ax, x_sig, y_sig, msz_sq, c_sig, 's', ...
                'filled', 'MarkerFaceColor','flat', 'MarkerEdgeColor','k', 'LineWidth', lw_filled);
        end
        if ~isempty(x_nsig)
            scatter(ax, x_nsig, y_nsig, msz_sq, c_nsig, 's', ...
                'MarkerFaceColor','none', 'MarkerEdgeColor','flat', 'LineWidth', lw_hollow);
        end

        % connect significant points
        if numel(idx_sig) >= 2
            [~, ord] = sort(idx_sig, 'ascend');
            ln = plot(ax, x_sig(ord), y_sig(ord), 'k-', 'LineWidth', 1.5);
            try, ln.Color(4) = lineAlphaBand; catch, end
        end

        % ===== title + fit + R2 (MOVED DOWN) =====
        xl = xlims; yl = ylims_shared;
        xText  = xl(1) + 0.02*(xl(2)-xl(1));

        % moved DOWN and separated to avoid overlap among title, fit equation, and R^2 text
        yTitle = yl(2) - 0.055*(yl(2)-yl(1));
        yEq    = yl(2) - 0.170*(yl(2)-yl(1));
        yR2    = yl(2) - 0.315*(yl(2)-yl(1));

        titleStr = sprintf('%s, Str = %.3g', names_cols{cIdx}, strList(rIdx));
        text(ax, xText, yTitle, titleStr, ...
            'FontName',fontName,'FontSize',fontSize,'FontWeight','bold', ...
            'BackgroundColor','w','Margin',3);

        x_all = avgCS(base_mask);  y_all = yPerf(base_mask);
        x_all = x_all(:);          y_all = y_all(:);

        okFit = (numel(x_all) >= 2) && isfinite(range(x_all)) && (range(x_all) > 0);
        if okFit
            pp = polyfit(x_all, y_all, 1);
            m  = pp(1);
            b0 = pp(2);

            lo = max(xlims(1), min(x_all));
            hi = min(xlims(2), max(x_all));
            if hi > lo
                xfit = linspace(lo, hi, 200);
                yfit = polyval(pp, xfit);
                plot(ax, xfit, yfit, 'r--', 'LineWidth', 2);
            end

            yhat  = polyval(pp, x_all);
            SSres = sum((y_all - yhat).^2);
            SStot = sum((y_all - mean(y_all)).^2);
            if SStot > 0, R2 = 1 - SSres/SStot; else, R2 = NaN; end

            if b0 >= 0
                eqStr = sprintf('y = %.3g x + %.3g', m, b0);
            else
                eqStr = sprintf('y = %.3g x - %.3g', m, abs(b0));
            end

            text(ax, xText, yEq, eqStr, ...
                'FontName',fontName,'FontSize',fontSize, ...
                'BackgroundColor','w','Margin',3);

            if isfinite(R2)
                text(ax, xText, yR2, sprintf('R^2_{fit} = %.3f', R2), ...
                    'FontName',fontName,'FontSize',R2fitFontSize, ...
                    'FontWeight','bold','BackgroundColor','w','Margin',3);
            else
                text(ax, xText, yR2, 'R^2_{fit} undefined (SStot=0)', ...
                    'FontName',fontName,'FontSize',R2fitFontSize, ...
                    'FontWeight','bold','BackgroundColor','w','Margin',3);
            end
        else
            text(ax, xText, yEq, 'Fit skipped (insufficient points)', ...
                'FontName',fontName,'FontSize',fontSize, ...
                'BackgroundColor','w','Margin',3);
        end

        % axes limits/ticks
        ylim(ax, ylims_shared);
        xlim(ax, xlims);
        set(ax,'XTick',0:0.2:1);
        set(ax,'XMinorTick','on');
        grid(ax,'off');
        caxis(ax,[0 1]);

        % ===== labels (your new request) =====
        % keep middle labels (bold)
        if (rIdx==2) && (cIdx==2)
            xlabel(ax, 'CS_{MDEA}', 'FontName',fontName,'FontSize',labelFontSize,'FontWeight','bold');
            ylabel(ax, 'C_r', 'FontName',fontName,'FontSize',labelFontSize,'FontWeight','bold');
        else
            xlabel(ax,'');
            ylabel(ax,'');
        end

        % add y-label for left panel in middle row
        if (rIdx==2) && (cIdx==1)
            ylabel(ax, 'C_r', ...
                'FontName',fontName,'FontSize',labelFontSize,'FontWeight','bold');
        end

        % add x-label for middle panel in third row
        if (rIdx==3) && (cIdx==2)
            xlabel(ax, 'CS_{MDEA}', ...
                'FontName',fontName,'FontSize',labelFontSize,'FontWeight','bold');
        end

        % y-axis removal rules you liked before
        if (cIdx==2) && (rIdx>1)
            set(ax,'YTickLabel',[],'YColor','none');
        end
        if cIdx>1
            set(ax,'YTickLabel',[]);
        end

        hold(ax,'off');
    end
end

drawnow;
for r=1:nStr
    for c=1:3
        ax = axs(r,c);
        ax.Position   = ax.OuterPosition;
        ax.LooseInset = [0 0 0 0];
    end
end
drawnow;

% colorbar (same height as top-right panel) + moved DOWN a bit
axTR = axs(1,3);
cb = colorbar(axTR);
cb.Label.String   = 'r';
cb.Label.FontName = fontName;
cb.Label.FontSize = labelFontSize;

nt = min(7, ry);
tickIdx = round(linspace(1, ry, nt));
cb.Ticks = (tickIdx - 1) / max(ry-1,1);
cb.TickLabels = compose('%.2f', RS(tickIdx));

posTR = axTR.Position;
cbW   = 0.020;
gap   = 0.015;

yShiftDown = 0.020; % <-- move gray bar down a little
yCB = max(0.01, posTR(2) - yShiftDown);

cb.Position = [posTR(1) + posTR(3) + gap, yCB, cbW, posTR(4)];



toc
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

    if Rule == 0
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
        xlabel('t'), ylabel('X(t)');
        legend('X(t)', 'Location', 'northwest');
        title('Signal');

        subplot(1, 2, 2)
        plot(de(3:length(de)-3), DE(3:length(DE)-3), '+', de0, FitLine(1)*de0 + FitLine(2), 'r--', 'LineWidth', 1.5);
        xlabel('log(l)'), ylabel('S(l)');
        legend(['\delta = ' num2str(sprintf('%.3f', delta))], 'Location', 'northwest');
    end
end
