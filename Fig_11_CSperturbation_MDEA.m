% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 11: MDEA-based CS under payoff-learning perturbation.
%
% What this script does:
%   - Sweeps perturbation strength k applied to the payoff-sharing adaptive channel.
%   - Computes MDEA-based CS for all 15 unique threshold pairs among [I1,T1,P1,I2,T2,P2].
%   - Highlights pairs involving payoff-sharing thresholds P1 or P2 and plots other pairs in the background.
%   - Rows are R = 0.25 and R = 0.45; columns are stripe sizes 0.1, 0.01, and 0.001.
%   - Parallel Computing Toolbox is used if available. This script currently uses a fixed baseSeed for reproducible sweeps.
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
clear
close all

%% ========================= USER PARAMETERS =========================
RRR    = 1;          % 1 = compute CS block
Time   = 1e6;        % number of trials
L      = 2;          % 2D environment size
dt     = 1;          % time step
deltaS = 0.1;        % base learning step

RS_list = [0.25, 0.45];
nRS     = numel(RS_list);

Noise   = 1e-3;      % base noise on threshold updates
strList = [1e-1, 1e-2, 1e-3];
nStr    = numel(strList);

EnvSuccessProb = 1;

kList = 0.01:0.0025:0.1;
nK    = numel(kList);

nEns     = 10;
baseSeed = 12345;     % fixed seed for reproducible perturbation sweeps

p_thr = 0.05;

% Channel labels in CONSISTENT order
chan_labels = {'I1','T1','P1','I2','T2','P2'};
nChan       = numel(chan_labels);

% All unique unordered pairs (6 choose 2 = 15)
pair_idx   = nchoosek(1:nChan, 2);
nPairs     = size(pair_idx, 1);
pair_names = cell(1, nPairs);
for pp = 1:nPairs
    pair_names{pp} = sprintf('%s__%s', chan_labels{pair_idx(pp,1)}, chan_labels{pair_idx(pp,2)});
end

% Identify pairs involving SharePay thresholds P1 or P2
isSharePayPair = (pair_idx(:,1) == 3) | (pair_idx(:,2) == 3) | ...
                 (pair_idx(:,1) == 6) | (pair_idx(:,2) == 6);

highlight_idx  = find(isSharePayPair);
background_idx = find(~isSharePayPair);

% Colors
highlight_colors = lines(numel(highlight_idx));
ccColor = [0.15 0.55 0.15];

%% ========================= OUTPUT ARRAYS =========================
CC_all  = nan(nEns, nK, nRS);
CS_all  = nan(nEns, nK, nPairs, nStr, nRS);
pCS_all = nan(nEns, nK, nPairs, nStr, nRS);

%% ========================= PARALLEL POOL =========================
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

%% ========================= PROGRESS BAR FOR PARFOR =========================
nTotalRuns = nEns * nK * nRS;

hWait = [];
try
    hWait = waitbar(0, 'Running perturbation sweep (0%)', 'Name', 'Simulation Progress');
catch
    hWait = [];
end

D = parallel.pool.DataQueue;
afterEach(D, @localUpdateProgress);

%% ========================= MAIN SWEEP =========================
parfor ee = 1:nEns

    CC_row   = nan(nK, nRS);
    CS_row   = nan(nK, nPairs, nStr, nRS);
    pCS_row  = nan(nK, nPairs, nStr, nRS);

    for rr = 1:nRS

        RSfac = RS_list(rr);

        for kk = 1:nK

            kPay = kList(kk);
            rng(baseSeed + 1000*ee + 100*rr + kk, 'twister');

            % ---------------- single-run params ----------------
            ThetaDecept = pi/6;

            rS   = RSfac * L;
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

            PayF0 = zeros(1, 1); 
            PayF  = zeros(1, 1); 
            PayS0 = zeros(2, 1);
            PayS  = zeros(2, 1);

            ShareInfoS1 = ones(Time, 1);
            ShareInfoS2 = ones(Time, 1);
            TrustS1     = ones(Time, 1);
            TrustS2     = ones(Time, 1);
            SharePayS1  = ones(Time, 1);
            SharePayS2  = ones(Time, 1);

            DecInfoS1   = zeros(Time, 1);
            DecInfoS2   = zeros(Time, 1);
            DecTrustS1  = zeros(Time, 1);
            DecTrustS2  = zeros(Time, 1);

            PayS1 = zeros(Time, 1);
            PayS2 = zeros(Time, 1);

            xF  = L*rand(1, 1);  yF = L*rand(1, 1);
            xS  = L*rand(1, 2);  yS = L*rand(1, 2);
            xFS = zeros(1, 3);
            yFS = zeros(1, 3);

            Ratio_CC  = zeros(Time, 1);
            Ratio_Pay = zeros(Time, 1); 
            CC = 0;
            TotalPay = 0;

            % ---------------- time loop ----------------
            for ti = 2:Time

                xFS(1) = xF(1);     yFS(1) = yF(1);
                xFS(2) = xS(1, 1);  yFS(2) = yS(1, 1);
                xFS(3) = xS(1, 2);  yFS(3) = yS(1, 2);
                thetaFS(1) = thetaF(1);
                thetaFS(2) = thetaS(1);
                thetaFS(3) = thetaS(2);

                % ---------- F angles ----------
                [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rF, L);
                list = l1FS(l2FS == 1);
                list = list(list > 1);
                if ~isempty(list)
                    xSthatFpredicts = mean(xFS(list)) + velS*mean(cos(thetaFS(list)))*dt;
                    ySthatFpredicts = mean(yFS(list)) + velS*mean(sin(thetaFS(list)))*dt;
                    tet1            = AnglePeriodic_torus(xSthatFpredicts, ySthatFpredicts, xFS(1), yFS(1), L);
                    thetaDeflect    = (ThetaDecept) * sign(-1 + 2*rand);
                    thetaFoeF(1)       = pi + tet1 + thetaDeflect;
                    thetaFoeFDECEPT(1) = pi + tet1 - thetaDeflect;
                else
                    thetaFoeF(1)       = thetaFS(1);
                    thetaFoeFDECEPT(1) = thetaFS(1);
                end

                if rand < 0.5
                    thetaF(1) = thetaFoeF(1);
                else
                    thetaF(1) = thetaFoeFDECEPT(1);
                end

                % ---------- S angles ----------
                [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rS, L);

                list = l1FS(l2FS == 2);
                list = list(list <= 1);
                if ~isempty(list)
                    xFthatSpredicts       = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
                    yFthatSpredicts       = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
                    xFthatSpredictsDECEPT = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
                    yFthatSpredictsDECEPT = mean(yFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;
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

                list = l1FS(l2FS == 3);
                list = list(list <= 1);
                if ~isempty(list)
                    xFthatSpredicts       = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
                    yFthatSpredicts       = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
                    xFthatSpredictsDECEPT = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
                    yFthatSpredictsDECEPT = mean(yFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;
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

                Trusted1 = 0;
                if rand > TrustS1(ti)
                    Trusted1 = 1;
                    thetaS(1) = thetaFoeSShared(2);
                end

                Trusted2 = 0;
                if rand > TrustS2(ti)
                    Trusted2 = 1;
                    thetaS(2) = thetaFoeSShared(1);
                end

                xF = mod(xF + velF * cos(thetaF) * dt, L);
                yF = mod(yF + velF * sin(thetaF) * dt, L);
                xS = mod(xS + velS * cos(thetaS) * dt, L);
                yS = mod(yS + velS * sin(thetaS) * dt, L);

                xFS = [xF(1), xS(1,1), xS(1,2)];
                yFS = [yF(1), yS(1,1), yS(1,2)];
                [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rG, L);

                list = l1FS(l2FS == 2);
                listofF = list(list <= 1);
                LF1 = length(listofF);

                list = l1FS(l2FS == 3);
                listofF = list(list <= 1);
                LF2 = length(listofF);

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

                if zPayS1 == -13, PayS(1) = -2; else, PayS(1) = 2; end
                if zPayS2 == -13, PayS(2) = -2; else, PayS(2) = 2; end

                any_win = ((zPayS1 == 13) || (zPayS2 == 13));

                s1 = (rand > SharePayS1(ti));
                s2 = (rand > SharePayS2(ti));

                ShareUSedS1 = 0;
                ShareUSedS2 = 0;

                if Trusted1 == 1 || Trusted2 == 1
                    if any_win
                        ShareUSedS1 = 1;
                        ShareUSedS2 = 1;

                        if s1 && s2
                            PayS = [2, 2];
                            CC = CC + 1;
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

                ShareInfoS1(ti + 1) = UpdateThreshold(1, ShareInfoS1(ti), ShaInfo1, PayS(1), PayS0(1), deltaS, Noise);
                ShareInfoS2(ti + 1) = UpdateThreshold(1, ShareInfoS2(ti), ShaInfo2, PayS(2), PayS0(2), deltaS, Noise);
                TrustS1(ti + 1)     = UpdateThreshold(1, TrustS1(ti), Trusted1, PayS(1), PayS0(1), deltaS, Noise);
                TrustS2(ti + 1)     = UpdateThreshold(1, TrustS2(ti), Trusted2, PayS(2), PayS0(2), deltaS, Noise);

                SharePayS1(ti + 1)  = UpdateThreshold(ShareUSedS1, SharePayS1(ti), s1, PayS(1), PayS0(1), kPay, Noise);
                SharePayS2(ti + 1)  = UpdateThreshold(ShareUSedS2, SharePayS2(ti), s2, PayS(2), PayS0(2), kPay, Noise);

                PayS0 = PayS;

                TotalPay      = TotalPay + mean(PayS);
                Ratio_Pay(ti) = TotalPay / ti;
                Ratio_CC(ti)  = CC / ti;

                PayS1(ti) = PayS(1);
                PayS2(ti) = PayS(2);

                DecInfoS1(ti)  = ShaInfo1;
                DecInfoS2(ti)  = ShaInfo2;
                DecTrustS1(ti) = Trusted1;
                DecTrustS2(ti) = Trusted2;
            end

            CC_row(kk,rr) = Ratio_CC(end);

            if RRR == 1
                ST0 = floor(0.25 * length(SharePayS1));
                EN0 = length(SharePayS1) - 1;

                data = zeros(EN0 - ST0 + 1, 6);
                data(:,1) = ShareInfoS1(ST0:EN0);
                data(:,2) = TrustS1(    ST0:EN0);
                data(:,3) = SharePayS1( ST0:EN0);
                data(:,4) = ShareInfoS2(ST0:EN0);
                data(:,5) = TrustS2(    ST0:EN0);
                data(:,6) = SharePayS2( ST0:EN0);

                Slice   = 1e4;
                Overlap = floor(0.75 * Slice);
                Newdata = Slice - Overlap;
                nn      = max(0, floor((size(data,1) - Slice) / Newdata));

                if nn > 0
                    for ss = 1:nStr

                        local_CORRYCS  = zeros(6, 6);
                        local_CORRYCSp = ones(6, 6);
                        Scale          = zeros(max(nn,1), 6);
                        str            = strList(ss);

                        for hh11 = 1:6
                            for gg11 = 1:nn
                                sta   = (gg11 - 1) * Newdata;
                                DaTaa = data(1 + sta : Slice + sta, hh11);

                                DataX = DaTaa - min(DaTaa);
                                if max(DataX) ~= 0
                                    DataX = DataX ./ max(DataX);
                                    Scale(gg11, hh11) = MDEA(DataX, str, 1, 0.1, 0.9, 0);
                                else
                                    Scale(gg11, hh11) = 0;
                                end
                            end
                        end

                        for ii = 1:6
                            for jj = 1:6
                                [a, p] = corrcoef(Scale(:,ii), Scale(:,jj));
                                local_CORRYCS(ii,jj)  = a(2,1);
                                local_CORRYCSp(ii,jj) = p(2,1);
                            end
                        end

                        for pp = 1:nPairs
                            i1 = pair_idx(pp,1);
                            i2 = pair_idx(pp,2);
                            CS_row(kk,pp,ss,rr)  = local_CORRYCS(i1,i2);
                            pCS_row(kk,pp,ss,rr) = local_CORRYCSp(i1,i2);
                        end
                    end
                else
                    for ss = 1:nStr
                        CS_row(kk,:,ss,rr)  = 0;
                        pCS_row(kk,:,ss,rr) = 1;
                    end
                end
            end

            send(D, 1);
        end
    end

    CC_all(ee,:,:)      = CC_row;
    CS_all(ee,:,:,:,:)  = CS_row;
    pCS_all(ee,:,:,:,:) = pCS_row;
end

try
    if ~isempty(hWait) && ishandle(hWait)
        waitbar(1, hWait, 'Done (100%)');
        pause(0.05);
        close(hWait);
    end
catch
end

%% ========================= SUMMARY STATS =========================
CC_mean  = squeeze(mean(CC_all, 1, 'omitnan'));
CC_std   = squeeze(std(CC_all,  0, 1, 'omitnan'));

CS_mean  = squeeze(mean(CS_all, 1, 'omitnan'));
CS_std   = squeeze(std(CS_all,  0, 1, 'omitnan'));
pCS_mean = squeeze(mean(pCS_all,1, 'omitnan'));

%% ========================= PLOTTING: 2x3 PANELS =========================
fontName      = 'Arial';
fontSize      = 12;
labelFontSize = 14;
infoFontSize  = 12;

lw_mean  = 2.4;
lw_bound = 0.8;   % weaker band boundaries

msz_sq   = round(0.85 * 55);

lineAlpha    = 0.78;
faceAlpha_sq = 0.78;
edgeAlpha_sq = 1.00;

bandAlpha = 0.95;   % lighter so they do not cover curves
bandTint  = 0.10;
edgeTint  = 0.08;
edgeAlpha = 0.30;   % weaker band boundary lines

alphaColor = @(c,a) (1-a)*[1 1 1] + a*c;

xlims = [min(kList) max(kList)];
ylims_shared = [-0.5 1];

fig = figure('Units','pixels','Position',[40 40 1850 930], 'Color','w');
set(fig,'Renderer','opengl');

tl = tiledlayout(nRS, nStr);
tl.TileSpacing = 'none';
tl.Padding     = 'none';
tl.Position    = [0.055 0.08 0.78 0.86];

axs = gobjects(nRS,nStr);
hLeg   = gobjects(numel(highlight_idx)+2,1);
axLast = [];


xTextFrac = 0.03;
yTitleFrac = 0.12;
ySubFrac   = 0.22;

for rr = 1:nRS
    for ss = 1:nStr
        ax = nexttile;
        axs(rr,ss) = ax;
        axLast = ax;
        hold(ax,'on'); box(ax,'on');

        set(ax,'FontName',fontName,'FontSize',fontSize, ...
            'Units','normalized', ...
            'PositionConstraint','outerposition', ...
            'ActivePositionProperty','outerposition', ...
            'LooseInset',[0 0 0 0], ...
            'LineWidth',1.2);

        axBG = ax.Color;

        CSm = CS_mean(:,:,ss,rr);
        CSs = CS_std(:,:,ss,rr);
        CPm = pCS_mean(:,:,ss,rr);
        CCm = CC_mean(:,rr);
        CCs = CC_std(:,rr);

        %% ===== DRAW ALL BANDS FIRST =====

        % Background pair bands
        for bb = 1:numel(background_idx)
            pp = background_idx(bb);

            c   = [0.38 0.38 0.38];
            cBd = alphaColor(c, edgeTint);
            cFi = alphaColor(c, bandTint);

            avgY = CSm(:,pp);
            sdY  = CSs(:,pp);

            base_mask = isfinite(kList(:)) & isfinite(avgY) & isfinite(sdY);
            x = kList(base_mask); x = x(:);
            y = avgY(base_mask);  y = y(:);
            s = sdY(base_mask);   s = s(:);

            yu = y + s;
            yl = y - s;

            [x, ord] = sort(x);
            yu = yu(ord);
            yl = yl(ord);

            if numel(x) >= 2
                fill(ax, [x; flipud(x)], [yu; flipud(yl)], cFi, ...
                    'FaceAlpha', bandAlpha, 'EdgeColor', 'none');
                up = plot(ax, x, yu, '-', 'Color', cBd, 'LineWidth', lw_bound);
                lo = plot(ax, x, yl, '-', 'Color', cBd, 'LineWidth', lw_bound);
                try, up.Color(4) = edgeAlpha; lo.Color(4) = edgeAlpha; end
            end
        end

        % Highlight pair bands
        for hh = 1:numel(highlight_idx)
            pp = highlight_idx(hh);
            c  = highlight_colors(hh,:);
            cBd = alphaColor(c, edgeTint);
            cFi = alphaColor(c, bandTint);

            avgY = CSm(:,pp);
            sdY  = CSs(:,pp);

            base_mask = isfinite(kList(:)) & isfinite(avgY) & isfinite(sdY);
            x = kList(base_mask); x = x(:);
            y = avgY(base_mask);  y = y(:);
            s = sdY(base_mask);   s = s(:);

            yu = y + s;
            yl = y - s;

            [x, ord] = sort(x);
            yu = yu(ord);
            yl = yl(ord);

            if numel(x) >= 2
                fill(ax, [x; flipud(x)], [yu; flipud(yl)], cFi, ...
                    'FaceAlpha', bandAlpha, 'EdgeColor', 'none');
                up = plot(ax, x, yu, '-', 'Color', cBd, 'LineWidth', lw_bound);
                lo = plot(ax, x, yl, '-', 'Color', cBd, 'LineWidth', lw_bound);
                try, up.Color(4) = edgeAlpha; lo.Color(4) = edgeAlpha; end
            end
        end

        % CC band
        c   = ccColor;
        cFi = alphaColor(c, bandTint);
        cBd = alphaColor(c, edgeTint);

        x = kList(:);
        y = CCm(:);
        s = CCs(:);

        base_mask = isfinite(x) & isfinite(y) & isfinite(s);
        x = x(base_mask); y = y(base_mask); s = s(base_mask);

        yu = y + s;
        yl = y - s;

        [x, ord] = sort(x);
        yu = yu(ord);
        yl = yl(ord);

        if numel(x) >= 2
            fill(ax, [x; flipud(x)], [yu; flipud(yl)], cFi, ...
                'FaceAlpha', bandAlpha, 'EdgeColor', 'none');
            up = plot(ax, x, yu, '-', 'Color', cBd, 'LineWidth', lw_bound);
            lo = plot(ax, x, yl, '-', 'Color', cBd, 'LineWidth', lw_bound);
            try, up.Color(4) = edgeAlpha; lo.Color(4) = edgeAlpha; end
        end

        %% ===== DRAW MEAN CURVES AND MARKERS ON TOP =====

        % Background pairs
        for bb = 1:numel(background_idx)
            pp = background_idx(bb);

            c   = [0.38 0.38 0.38];
            cLn = alphaColor(c, lineAlpha);

            avgY = CSm(:,pp);
            avgP = CPm(:,pp);

            base_mask = isfinite(kList(:)) & isfinite(avgY) & isfinite(avgP);
            sig_mask  = base_mask & (avgP < p_thr);
            nsig_mask = base_mask & ~sig_mask;

            x = kList(base_mask); x = x(:);
            y = avgY(base_mask);  y = y(:);

            [x, ord] = sort(x);
            y = y(ord);

            plot(ax, x, y, '-', 'LineWidth', lw_mean-0.4, 'Color', cLn);

            if any(sig_mask)
                xs = kList(sig_mask);
                ys = avgY(sig_mask);
                scatter(ax, xs, ys, msz_sq*0.78, 's','filled', ...
                    'MarkerFaceColor', c, 'MarkerEdgeColor', c, ...
                    'MarkerFaceAlpha', faceAlpha_sq, ...
                    'MarkerEdgeAlpha', edgeAlpha_sq);
            end

            if any(nsig_mask)
                xn = kList(nsig_mask);
                yn = avgY(nsig_mask);

                scatter(ax, xn, yn, msz_sq*0.78, 's','filled', ...
                    'MarkerFaceColor', axBG, 'MarkerEdgeColor', axBG, ...
                    'MarkerFaceAlpha', faceAlpha_sq, ...
                    'MarkerEdgeAlpha', edgeAlpha_sq);

                c_edge = alphaColor(c, 1-edgeTint);
                hTmpBG = scatter(ax, xn, yn, msz_sq*0.78, 's', ...
                    'MarkerFaceColor','none', ...
                    'MarkerEdgeColor', c_edge, ...
                    'LineWidth', 2.1, ...
                    'MarkerEdgeAlpha', edgeAlpha_sq);
                if rr == nRS && ss == nStr && bb == 1
                    hLeg(end) = hTmpBG;
                end
            elseif rr == nRS && ss == nStr && bb == 1
                hLeg(end) = plot(ax, nan, nan, '-', ...
                    'LineWidth', lw_mean-0.4, 'Color', cLn, ...
                    'Marker','s', ...
                    'MarkerSize', sqrt(msz_sq*0.78), ...
                    'MarkerFaceColor','none', ...
                    'MarkerEdgeColor', c);
            end
        end

        % Highlighted pairs
        for hh = 1:numel(highlight_idx)
            pp = highlight_idx(hh);
            c  = highlight_colors(hh,:);
            cLn = alphaColor(c, lineAlpha);

            avgY = CSm(:,pp);
            avgP = CPm(:,pp);

            base_mask = isfinite(kList(:)) & isfinite(avgY) & isfinite(avgP);
            sig_mask  = base_mask & (avgP < p_thr);
            nsig_mask = base_mask & ~sig_mask;

            x = kList(base_mask); x = x(:);
            y = avgY(base_mask);  y = y(:);

            [x, ord] = sort(x);
            y = y(ord);

            plot(ax, x, y, '-', 'LineWidth', lw_mean, 'Color', cLn);

            if any(sig_mask)
                xs = kList(sig_mask);
                ys = avgY(sig_mask);

                scSig = scatter(ax, xs, ys, msz_sq, 's','filled', ...
                    'MarkerFaceColor', c, 'MarkerEdgeColor', c, ...
                    'MarkerFaceAlpha', faceAlpha_sq, ...
                    'MarkerEdgeAlpha', edgeAlpha_sq);
            else
                scSig = plot(ax, nan, nan, '-', ...
                    'LineWidth', lw_mean, 'Color', cLn, ...
                    'Marker','s', ...
                    'MarkerSize', sqrt(msz_sq), ...
                    'MarkerFaceColor', c, ...
                    'MarkerEdgeColor', c);
            end

            if any(nsig_mask)
                xn = kList(nsig_mask);
                yn = avgY(nsig_mask);

                scatter(ax, xn, yn, msz_sq, 's','filled', ...
                    'MarkerFaceColor', axBG, 'MarkerEdgeColor', axBG, ...
                    'MarkerFaceAlpha', faceAlpha_sq, ...
                    'MarkerEdgeAlpha', edgeAlpha_sq);

                c_edge = alphaColor(c, 1-edgeTint);
                scatter(ax, xn, yn, msz_sq, 's', ...
                    'MarkerFaceColor','none', ...
                    'MarkerEdgeColor', c_edge, ...
                    'LineWidth', 2.3, ...
                    'MarkerEdgeAlpha', edgeAlpha_sq);
            end

            if rr == nRS && ss == nStr
                hLeg(hh) = scSig;
            end
        end

        % CC dashed curve on top
        c   = ccColor;
        cLn = alphaColor(c, lineAlpha);

        x = kList(:);
        y = CCm(:);
        base_mask = isfinite(x) & isfinite(y);
        x = x(base_mask); y = y(base_mask);

        [x, ord] = sort(x);
        y = y(ord);

        hCC = plot(ax, x, y, '--', 'LineWidth', 2.8, 'Color', cLn);
        if rr == nRS && ss == nStr
            hLeg(numel(highlight_idx)+1) = hCC;
        end

        %% ===== TITLES INSIDE PANELS (BOTTOM-LEFT) =====
        xl = xlims; yl = ylims_shared;
        xText  = xl(1) + xTextFrac*(xl(2)-xl(1));
        yTitle = yl(1) + yTitleFrac*(yl(2)-yl(1));
        ySub   = yl(1) + ySubFrac*(yl(2)-yl(1));

        text(ax, xText, ySub, sprintf('R = %.2f', RS_list(rr)), ...
            'FontName',fontName, 'FontSize',infoFontSize, 'FontWeight','bold', ...
            'BackgroundColor','w', 'Margin',2);

        text(ax, xText, yTitle, sprintf('Stripe size = %.3g', strList(ss)), ...
            'FontName',fontName, 'FontSize',infoFontSize, 'FontWeight','bold', ...
            'BackgroundColor','w', 'Margin',2);

        %% ===== AXES =====
        ylim(ax, ylims_shared);
        xlim(ax, xlims);
        grid(ax,'off');

        if rr == nRS
            xlabel(ax, 'k', ...
                'FontName',fontName, ...
                'FontSize',14, ...
                'FontWeight','bold');
        else
            xlabel(ax,'');
            set(ax,'XTickLabel',[]);
        end

        if ss == 1
            ylabel(ax, 'CS_{MDEA} and C_{r}', ...
                'FontName',fontName, ...
                'FontSize',14, ...
                'FontWeight','bold');
        else
            ylabel(ax,'');
            set(ax,'YTickLabel',[]);
        end

        hold(ax,'off');
    end
end

drawnow;
for r = 1:nRS
    for c = 1:nStr
        ax = axs(r,c);
        ax.Position   = ax.OuterPosition;
        ax.LooseInset = [0 0 0 0];
    end
end
drawnow;

pair_names_leg = strrep(pair_names(highlight_idx),'__','-');
legend(axLast, hLeg, [pair_names_leg(:); {'C_{r}'}; {'other pairs'}], ...
    'Location','eastoutside', ...
    'FontSize',12);

sgtitle('CS_{MDEA} and C_{r} vs perturbation strength k', ...
    'FontName',fontName, ...
    'FontSize',17, ...
    'FontWeight','bold');

%% ========================= SAVE RESULTS =========================
ResultsPerturbationMDEA.kList          = kList;
ResultsPerturbationMDEA.RS_list        = RS_list;
ResultsPerturbationMDEA.deltaS         = deltaS;
ResultsPerturbationMDEA.Noise          = Noise;
ResultsPerturbationMDEA.strList        = strList;
ResultsPerturbationMDEA.nEns           = nEns;
ResultsPerturbationMDEA.p_thr          = p_thr;

ResultsPerturbationMDEA.chan_labels    = chan_labels;
ResultsPerturbationMDEA.pair_idx       = pair_idx;
ResultsPerturbationMDEA.pair_names     = pair_names;
ResultsPerturbationMDEA.isSharePayPair = isSharePayPair;
ResultsPerturbationMDEA.highlight_idx  = highlight_idx;
ResultsPerturbationMDEA.background_idx = background_idx;

ResultsPerturbationMDEA.CC_all         = CC_all;
ResultsPerturbationMDEA.CC_mean        = CC_mean;
ResultsPerturbationMDEA.CC_std         = CC_std;

ResultsPerturbationMDEA.CS_all         = CS_all;
ResultsPerturbationMDEA.pCS_all        = pCS_all;
ResultsPerturbationMDEA.CS_mean        = CS_mean;
ResultsPerturbationMDEA.CS_std         = CS_std;
ResultsPerturbationMDEA.pCS_mean       = pCS_mean;

save('PerturbationSweep_CS_MDEA_CC_allPairs_twoRS_threeStripe_vs_k.mat', 'ResultsPerturbationMDEA');

toc

%% ========================= LOCAL FUNCTIONS =========================
function localUpdateProgress(~)
    persistent count h totalRuns
    if isempty(count)
        count = 0;
        h = evalin('base', 'hWait');
        totalRuns = evalin('base', 'nTotalRuns');
    end
    count = count + 1;
    try
        if ~isempty(h) && ishandle(h)
            frac = count / totalRuns;
            waitbar(frac, h, sprintf('Running perturbation sweep (%.1f%%%%)', 100*frac));
        end
    catch
    end
end

function theta = AnglePeriodic_torus(x_to, y_to, x_from, y_from, L)
    dx = x_to - x_from;
    dy = y_to - y_from;
    dx = dx - L*floor(dx/L + 0.5);
    dy = dy - L*floor(dy/L + 0.5);
    theta = atan2(dy, dx);
end

function [A, B] = Finddistance_torus(x, y, r, L)
    x = x(:)';
    y = y(:)';
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
        if Decision0 == 1
            dp = -1 * ChangeThreshold;
        else
            dp =  1 * ChangeThreshold;
        end

        if Pay ~= 0 && Paybefore ~= 0
            DeltaPay = (Pay - Paybefore);
        else
            DeltaPay = 0;
        end

        DeltaPay = DeltaPay + noiseInt * randn;
        aa = pi0 + dp * DeltaPay;
    else
        aa = pi0 + noiseInt * randn;
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
    StartEvent = zeros(Lengthdata,1);

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
                if rand < 0.5
                    State0(yy) = 1;
                else
                    State0(yy) = -1;
                end
            end
        end
        Diff = cumsum(State0);
    else
        State0 = zeros(Lengthdata, 1);
        for yy = 1:Lengthdata
            if Event(yy) == 1
                if rand < 0.5
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
        SliceNum = max(1, length(StartEvent));
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
        if isempty(XF)
            DE(:, q) = 0;
            continue
        end

        nbins = floor(max(abs(XF)) / 1);
        if isempty(nbins) || nbins <= 0
            DE(:, q) = 0;
            continue
        end

        counts = hist(XF, nbins);
        counts = counts(counts ~= 0);
        Pc = counts ./ sum(counts);

        DE0 = -sum((Pc) .* log(Pc) / log(10));
        DE(:, q) = DE0;
    end

    for t = 1:length(Delh)
        de(t) = log(Delh(t)) / log(10);
    end

    Starr = max(1, round(ST * length(de)));
    endd  = max(Starr, round(EN * length(de)));
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
        plot(de(3:max(3,length(de)-3)), DE(3:max(3,length(DE)-3)), '+'); hold on
        plot(de0, FitLine(1)*de0 + FitLine(2), 'r--', 'LineWidth', 1.5);
        xlabel('log(l)'), ylabel('S(l)');
        legend(['\delta = ' num2str(sprintf('%.3f', delta))], 'Location', 'northwest');
    end
end
