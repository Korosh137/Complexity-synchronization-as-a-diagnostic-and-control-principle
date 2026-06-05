% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 12: DFA-based CS under payoff-learning perturbation.
%
% What this script does:
%   - Sweeps perturbation strength k applied to the payoff-sharing adaptive channel.
%   - Computes DFA-based CS for all 15 unique threshold pairs among [I1,T1,P1,I2,T2,P2].
%   - Highlights pairs involving payoff-sharing thresholds P1 or P2 and plots other pairs in the background.
%   - Panels correspond to R = 0.25 and R = 0.45.
%   - Randomness: a shuffled base seed is generated each time the script is run.
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
EnvSuccessProb = 1;

kList = 0.01:0.0025:0.1;
nK    = numel(kList);

nEns     = 10;

% Use a different random base seed each time the script is run.
% This avoids repeating the exact same experiment across separate runs.
rng('shuffle', 'twister');
baseSeed = randi(1e9);

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
CS_all  = nan(nEns, nK, nPairs, nRS);
pCS_all = nan(nEns, nK, nPairs, nRS);

%% ========================= PARALLEL POOL =========================
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

%% ========================= PROGRESS BAR FOR PARFOR =========================
nTotalRuns = nEns * nK * nRS;

hWait = [];
try
    hWait = waitbar(0, 'Running DFA perturbation sweep (0%)', 'Name', 'Simulation Progress');
catch
    hWait = [];
end

D = parallel.pool.DataQueue;
afterEach(D, @localUpdateProgress);

%% ========================= MAIN SWEEP =========================
parfor ee = 1:nEns

    CC_row   = nan(nK, nRS);
    CS_row   = nan(nK, nPairs, nRS);
    pCS_row  = nan(nK, nPairs, nRS);

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

            PayF0 = zeros(1, 1); %#ok<NASGU>
            PayF  = zeros(1, 1); %#ok<NASGU>
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
            Ratio_Pay = zeros(Time, 1); %#ok<NASGU>
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

                local_CORRYCS  = zeros(6, 6);
                local_CORRYCSp = ones(6, 6);
                Scale          = zeros(max(nn,1), 6);

                if nn > 0
                    for hh11 = 1:6
                        for gg11 = 1:nn
                            sta   = (gg11 - 1) * Newdata;
                            DaTaa = data(1 + sta : Slice + sta, hh11);

                            DataX = DaTaa - min(DaTaa);
                            if max(DataX) ~= 0
                                aa = DFA_func(DaTaa, 100:100:1000, 1, 0);
                                Scale(gg11, hh11) = aa(1);
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
                        CS_row(kk,pp,rr)  = local_CORRYCS(i1,i2);
                        pCS_row(kk,pp,rr) = local_CORRYCSp(i1,i2);
                    end
                else
                    CS_row(kk,:,rr)  = 0;
                    pCS_row(kk,:,rr) = 1;
                end
            end

            send(D, 1);
        end
    end

    CC_all(ee,:,:)    = CC_row;
    CS_all(ee,:,:,:)  = CS_row;
    pCS_all(ee,:,:,:) = pCS_row;
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

%% ========================= PLOTTING: 1x2 PANELS =========================
fontName      = 'Arial';
fontSize       = 14;   % tick-label font size
labelFontSize  = 20;   % x/y-axis label font size
infoFontSize   = 13;   % inside-panel RS text font size
titleFontSize  = 19;   % main title font size
legendFontSize = 14;   % right-side legend font size

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

fig = figure('Units','pixels','Position',[40 120 1750 580], 'Color','w');
set(fig,'Renderer','opengl');

% Use tiledlayout-level legend placement.
% The legend will occupy its own dedicated tile on the right side of BOTH panels.
tl = tiledlayout(1, nRS);
tl.TileSpacing = 'compact';
tl.Padding     = 'compact';

axs = gobjects(1,nRS);
hLeg   = gobjects(numel(highlight_idx)+2,1);
axLast = [];


xTextFrac = 0.03;
ySubFrac   = 0.22;

for rr = 1:nRS
    ax = nexttile;
    axs(rr) = ax;
    axLast = ax;
    hold(ax,'on'); box(ax,'on');

    set(ax,'FontName',fontName,'FontSize',fontSize, ...
        'FontWeight','bold', ...
        'Units','normalized', ...
        'PositionConstraint','outerposition', ...
        'ActivePositionProperty','outerposition', ...
        'LooseInset',[0 0 0 0], ...
        'LineWidth',1.2);

    axBG = ax.Color;

    CSm = CS_mean(:,:,rr);
    CSs = CS_std(:,:,rr);
    CPm = pCS_mean(:,:,rr);
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
            try, up.Color(4) = edgeAlpha; lo.Color(4) = edgeAlpha; end %#ok<TRYNC>
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
            try, up.Color(4) = edgeAlpha; lo.Color(4) = edgeAlpha; end %#ok<TRYNC>
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
        try, up.Color(4) = edgeAlpha; lo.Color(4) = edgeAlpha; end %#ok<TRYNC>
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
            if rr == nRS && bb == 1
                hLeg(end) = hTmpBG;
            end
        elseif rr == nRS && bb == 1
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

        if rr == nRS
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
    if rr == nRS
        hLeg(numel(highlight_idx)+1) = hCC;
    end

    %% ===== TITLES INSIDE PANELS (BOTTOM-LEFT) =====
    xl = xlims; yl = ylims_shared;
    xText  = xl(1) + xTextFrac*(xl(2)-xl(1));
    ySub   = yl(1) + ySubFrac*(yl(2)-yl(1));

    text(ax, xText, ySub, sprintf('RS = %.2f', RS_list(rr)), ...
        'FontName',fontName, 'FontSize',infoFontSize, 'FontWeight','bold', ...
        'BackgroundColor','w', 'Margin',2);


    %% ===== AXES =====
    ylim(ax, ylims_shared);
    xlim(ax, xlims);
    grid(ax,'off');

    xlabel(ax, 'k', ...
        'FontName',fontName, ...
        'FontSize',labelFontSize, ...
        'FontWeight','bold');

    if rr == 1
        ylabel(ax, 'CS_{DFA} and C_{r}', ...
            'FontName',fontName, ...
            'FontSize',labelFontSize, ...
            'FontWeight','bold');
    else
        ylabel(ax,'');
        set(ax,'YTickLabel',[]);
    end

    hold(ax,'off');
end

drawnow;

pair_names_leg = strrep(pair_names(highlight_idx),'__','-');
legend_labels  = [pair_names_leg(:); {'C_{r}'}; {'other pairs'}];

% Put the legend outside the two panels, in its own tiledlayout column.
% This is different from 'eastoutside', which attaches the legend only to one axis.
lgd = legend(axLast, hLeg, legend_labels, ...
    'FontName',fontName, ...
    'FontSize',legendFontSize, ...
    'FontWeight','bold', ...
    'Box','off');
lgd.Layout.Tile = 'east';

sgtitle('CS_{DFA} and C_{r} vs perturbation strength k', ...
    'FontName',fontName, ...
    'FontSize',titleFontSize, ...
    'FontWeight','bold');

%% ========================= SAVE RESULTS =========================
ResultsPerturbationDFA.kList          = kList;
ResultsPerturbationDFA.RS_list        = RS_list;
ResultsPerturbationDFA.deltaS         = deltaS;
ResultsPerturbationDFA.Noise          = Noise;
ResultsPerturbationDFA.nEns           = nEns;
ResultsPerturbationDFA.p_thr          = p_thr;

ResultsPerturbationDFA.chan_labels    = chan_labels;
ResultsPerturbationDFA.pair_idx       = pair_idx;
ResultsPerturbationDFA.pair_names     = pair_names;
ResultsPerturbationDFA.isSharePayPair = isSharePayPair;
ResultsPerturbationDFA.highlight_idx  = highlight_idx;
ResultsPerturbationDFA.background_idx = background_idx;

ResultsPerturbationDFA.CC_all         = CC_all;
ResultsPerturbationDFA.CC_mean        = CC_mean;
ResultsPerturbationDFA.CC_std         = CC_std;

ResultsPerturbationDFA.CS_all         = CS_all;
ResultsPerturbationDFA.pCS_all        = pCS_all;
ResultsPerturbationDFA.CS_mean        = CS_mean;
ResultsPerturbationDFA.CS_std         = CS_std;
ResultsPerturbationDFA.pCS_mean       = pCS_mean;

save('PerturbationSweep_CS_DFA_CC_allPairs_twoRS_vs_k.mat', 'ResultsPerturbationDFA');

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
            waitbar(frac, h, sprintf('Running DFA perturbation sweep (%.1f%%%%)', 100*frac));
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

%%%  https://www.mathworks.com/matlabcentral/fileexchange/67889-detrended-fluctuation-analysis-dfa
function [A,F] = DFA_func(data, pts, order, PLOT)
% -----------------------------------------------------
% DESCRIPTION:
% Function for the DFA analysis.
% INPUTS:
% data: a one-dimensional data vector.
% pts: sizes of the windows/bins at which to evaluate the fluctuation
% order: order of the polynomial for the local trend correction.
% PLOT: 1 to plot, 0 otherwise
% OUTPUTS:
% A: 2x1 vector. A(1) is alpha, A(2) intercept
% F: fluctuations for each window size in pts
% -----------------------------------------------------
if nargin < 4
    PLOT = 0;
end
if nargin < 3 || isempty(order)
    order = 1;
end

sz = size(data);
if sz(1) < sz(2)
    data = data';
end

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
