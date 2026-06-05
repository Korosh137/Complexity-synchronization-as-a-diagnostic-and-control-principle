% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 18 / Supplementary control: raw Pearson correlations.
%
% What this script does:
%   - Computes ordinary signed Pearson correlations directly from the six raw adaptive threshold time series.
%   - Plots all 15 pairwise raw correlations against sensing-radius ratio R, with C_r shown as the foreground curve.
%   - This control separates raw signal correlation from complexity synchronization of scaling-exponent time series.
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

% ========================= PARAMETERS =========================
Time   = 1e6;       % use 1e6 for final manuscript runs
ENS    = 10;        % use 10 ensembles for manuscript-quality raw-pair error bands
L      = 2;
dt     = 1;
deltaS = 0.1;

RS_list = 0.25:0.01:0.45;
nRS     = numel(RS_list);

Noise = 1e-3;
EnvSuccessProb = 1;

ST0_frac = 0.25;

labels = {'I1','I2','T1','T2','P1','P2'};
nVar   = 6;

pairs  = nchoosek(1:nVar,2);
nPairs = size(pairs,1);

fprintf('\nAll 15 pairs used for raw correlation:\n');
for pp = 1:nPairs
    fprintf('%2d: %s-%s\n', pp, labels{pairs(pp,1)}, labels{pairs(pp,2)});
end


% ========================= SIGNIFICANCE SETTINGS =========================
p_thr = 0.05;       % raw Pearson significance threshold

% ========================= OUTPUTS =========================
CC_all          = zeros(ENS,nRS);
RawCorr_pairs_all = zeros(ENS,nRS,nPairs);
RawP_pairs_all    = NaN(ENS,nRS,nPairs);

% ========================= PARALLEL + PROGRESS =========================
if isempty(gcp('nocreate'))
    parpool;
end

seeds = randi(1e9,ENS,1);

chunkSize   = max(1,floor(Time/100));
ticksPerRun = ceil(Time/chunkSize);
totalTicks  = nRS * ENS * ticksPerRun;

dq = parallel.pool.DataQueue;
hWait = waitbar(0,'Starting simulations...','Name','Simulation Progress');
setappdata(hWait,'count',0);
setappdata(hWait,'total',totalTicks);
afterEach(dq,@(~) localIncrement(hWait));

% ========================= MAIN LOOP =========================
for rr = 1:nRS

    RS = RS_list(rr);

    parfor ens = 1:ENS

        rng(seeds(ens) + 100000*rr,'twister');

        rSfac       = RS;
        ThetaDecept = pi/6;

        rS   = rSfac * L;
        velS = 0.1;
        rF   = 1 * rS;
        velF = 2 * velS;
        rG   = 1 * rS;
        tc   = 3;

        thetaF          = 2*pi*(rand(1,1)-0.5);
        thetaFoeF       = zeros(1,1);
        thetaFoeFDECEPT = zeros(1,1);
        thetaS          = 2*pi*(rand(1,2)-0.5);
        thetaFS         = 2*pi*(rand(1,3)-0.5);
        thetaFoeSShared = zeros(1,2);

        PayS0 = zeros(2,1);
        PayS  = zeros(2,1);

        ShareInfoS1 = ones(Time+1,1);
        ShareInfoS2 = ones(Time+1,1);
        TrustS1     = ones(Time+1,1);
        TrustS2     = ones(Time+1,1);
        SharePayS1  = ones(Time+1,1);
        SharePayS2  = ones(Time+1,1);

        xF = L*rand(1,1);
        yF = L*rand(1,1);
        xS = L*rand(1,2);
        yS = L*rand(1,2);

        xFS = zeros(1,3);
        yFS = zeros(1,3);

        Ratio_CC = zeros(Time,1);
        CC = 0;

        % ========================= TIME LOOP =========================
        for ti = 2:Time

            xFS(1) = xF(1);     yFS(1) = yF(1);
            xFS(2) = xS(1,1);   yFS(2) = yS(1,1);
            xFS(3) = xS(1,2);   yFS(3) = yS(1,2);

            thetaFS(1) = thetaF(1);
            thetaFS(2) = thetaS(1);
            thetaFS(3) = thetaS(2);

            % ---------------- F angles ----------------
            [l1FS,l2FS] = Finddistance_torus(xFS,yFS,rF,L);
            list = l1FS(l2FS == 1);
            list = list(list > 1);

            if ~isempty(list)

                xSthatFpredicts = mean(xFS(list)) + velS*mean(cos(thetaFS(list)))*dt;
                ySthatFpredicts = mean(yFS(list)) + velS*mean(sin(thetaFS(list)))*dt;

                tet1 = AnglePeriodic_torus(xSthatFpredicts,ySthatFpredicts,xFS(1),yFS(1),L);

                thetaDeflect = ThetaDecept * sign(-1 + 2*rand);

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

            % ---------------- S1 angles ----------------
            [l1FS,l2FS] = Finddistance_torus(xFS,yFS,rS,L);

            list = l1FS(l2FS == 2);
            list = list(list <= 1);

            if ~isempty(list)

                xFthatSpredicts       = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
                yFthatSpredicts       = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
                xFthatSpredictsDECEPT = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
                yFthatSpredictsDECEPT = mean(yFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;

                if rand < 0.5

                    thetaS(1) = AnglePeriodic_torus(xFthatSpredicts,yFthatSpredicts,xFS(2),yFS(2),L);
                    thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredictsDECEPT,yFthatSpredictsDECEPT,xFS(3),yFS(3),L);

                else

                    thetaS(1) = AnglePeriodic_torus(xFthatSpredictsDECEPT,yFthatSpredictsDECEPT,xFS(2),yFS(2),L);
                    thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredicts,yFthatSpredicts,xFS(3),yFS(3),L);

                end

            else

                thetaS(1) = thetaFS(2);

            end

            % ---------------- S2 angles ----------------
            list = l1FS(l2FS == 3);
            list = list(list <= 1);

            if ~isempty(list)

                xFthatSpredicts       = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
                yFthatSpredicts       = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
                xFthatSpredictsDECEPT = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
                yFthatSpredictsDECEPT = mean(yFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;

                if rand < 0.5

                    thetaS(2) = AnglePeriodic_torus(xFthatSpredicts,yFthatSpredicts,xFS(3),yFS(3),L);
                    thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredictsDECEPT,yFthatSpredictsDECEPT,xFS(2),yFS(2),L);

                else

                    thetaS(2) = AnglePeriodic_torus(xFthatSpredictsDECEPT,yFthatSpredictsDECEPT,xFS(3),yFS(3),L);
                    thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredicts,yFthatSpredicts,xFS(2),yFS(2),L);

                end

            else

                thetaS(2) = thetaFS(3);

            end

            % ---------------- decisions ----------------
            ShaInfo1 = rand > ShareInfoS1(ti);
            ShaInfo2 = rand > ShareInfoS2(ti);

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

            % ---------------- integrate ----------------
            xF = mod(xF + velF*cos(thetaF)*dt,L);
            yF = mod(yF + velF*sin(thetaF)*dt,L);
            xS = mod(xS + velS*cos(thetaS)*dt,L);
            yS = mod(yS + velS*sin(thetaS)*dt,L);

            % ---------------- payoff neighborhood ----------------
            xFS = [xF(1), xS(1,1), xS(1,2)];
            yFS = [yF(1), yS(1,1), yS(1,2)];

            [l1FS,l2FS] = Finddistance_torus(xFS,yFS,rG,L);

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

            if zPayS1 == -13
                PayS(1) = -2;
            else
                PayS(1) = 2;
            end

            if zPayS2 == -13
                PayS(2) = -2;
            else
                PayS(2) = 2;
            end

            any_win = ((zPayS1 == 13) || (zPayS2 == 13));

            s1 = rand > SharePayS1(ti);
            s2 = rand > SharePayS2(ti);

            ShareUSedS1 = 0;
            ShareUSedS2 = 0;

            if Trusted1 == 1 || Trusted2 == 1

                if any_win

                    ShareUSedS1 = 1;
                    ShareUSedS2 = 1;

                    if s1 && s2

                        PayS = [2,2];
                        CC = CC + 1;

                    elseif ~s1 && s2

                        PayS = [2 + tc,-2];

                    elseif s1 && ~s2

                        PayS = [-2,2 + tc];

                    elseif ~s1 && ~s2

                        PayS = [-1,-1];

                    end

                end

            end

            ShareInfoS1(ti+1) = UpdateThreshold(1,ShareInfoS1(ti),ShaInfo1,PayS(1),PayS0(1),deltaS,Noise);
            ShareInfoS2(ti+1) = UpdateThreshold(1,ShareInfoS2(ti),ShaInfo2,PayS(2),PayS0(2),deltaS,Noise);
            TrustS1(ti+1)     = UpdateThreshold(1,TrustS1(ti),Trusted1,PayS(1),PayS0(1),deltaS,Noise);
            TrustS2(ti+1)     = UpdateThreshold(1,TrustS2(ti),Trusted2,PayS(2),PayS0(2),deltaS,Noise);

            SharePayS1(ti+1) = UpdateThreshold(ShareUSedS1,SharePayS1(ti),s1,PayS(1),PayS0(1),deltaS,Noise);
            SharePayS2(ti+1) = UpdateThreshold(ShareUSedS2,SharePayS2(ti),s2,PayS(2),PayS0(2),deltaS,Noise);

            PayS0 = PayS;

            Ratio_CC(ti) = CC / ti;

            if mod(ti,chunkSize) == 0 || ti == Time
                send(dq,1);
            end

        end

        % ========================= ANALYSIS =========================
        ST0 = floor(ST0_frac * Time);
        EN0 = Time;

        data = zeros(EN0-ST0+1,nVar);

        data(:,1) = ShareInfoS1(ST0:EN0);
        data(:,2) = ShareInfoS2(ST0:EN0);
        data(:,3) = TrustS1(ST0:EN0);
        data(:,4) = TrustS2(ST0:EN0);
        data(:,5) = SharePayS1(ST0:EN0);
        data(:,6) = SharePayS2(ST0:EN0);

        % ----- raw Pearson correlation + p-values -----
        % Rmat_raw gives the signed raw Pearson r values.
        % Pmat_raw gives the corresponding p-values testing H0: r = 0.
        % We DO NOT threshold the raw correlation values before computing
        % ensemble means. The p-values are used only for visual reliability
        % marking in the final figure.
        [Rmat_raw,Pmat_raw] = corrcoef(data,'Rows','pairwise');

        raw_vec = zeros(1,nPairs);
        raw_pvec = NaN(1,nPairs);
        for pp = 1:nPairs
            raw_vec(pp)  = Rmat_raw(pairs(pp,1),pairs(pp,2));
            raw_pvec(pp) = Pmat_raw(pairs(pp,1),pairs(pp,2));
        end

        CC_all(ens,rr) = Ratio_CC(end);

        RawCorr_pairs_all(ens,rr,:) = raw_vec;
        RawP_pairs_all(ens,rr,:)    = raw_pvec;

    end

end

% Close waitbar
try
    if ~isempty(hWait) && isvalid(hWait)
        waitbar(1,hWait,'Done');
        pause(0.05);
        close(hWait);
    end
catch
end

% ========================= ENSEMBLE SUMMARY =========================
CC_mean = mean(CC_all,1,'omitnan');
CC_std  = std(CC_all,0,1,'omitnan');


RawPair_mean = squeeze(mean(RawCorr_pairs_all,1,'omitnan'));
RawPair_std  = squeeze(std(RawCorr_pairs_all,0,1,'omitnan'));

% Fraction of ensembles in which each pairwise raw correlation is significant
% at each R value. This is used only for plotting reliability markers.
RawSigMask = double(RawP_pairs_all < p_thr);
RawSigMask(isnan(RawP_pairs_all)) = NaN;

RawPair_sigFrac = squeeze(mean(RawSigMask,1,'omitnan'));
RawPair_overallSigFrac = mean(RawSigMask(:),'omitnan');
RawPair_sigFrac_byRS = squeeze(mean(mean(RawSigMask,1,'omitnan'),3,'omitnan'));
RawPair_sigFrac_byPair = squeeze(mean(mean(RawSigMask,1,'omitnan'),2,'omitnan'));

fprintf('\n===== Summary across RS =====\n');
fprintf('RS range: %.2f to %.2f\n',min(RS_list),max(RS_list));
fprintf('Mean Cr range: %.4f to %.4f\n',min(CC_mean),max(CC_mean));
% Significance summaries are printed to the command window only; they are not
% shown in the figure legend.
fprintf('Raw Pearson significant tests: %.1f%%%% of pair x ensemble x R tests at p < %.3f\n',100*RawPair_overallSigFrac,p_thr);
fprintf('Raw Pearson significant fraction by R ranges from %.1f%%%% to %.1f%%%%\n',100*min(RawPair_sigFrac_byRS),100*max(RawPair_sigFrac_byRS));


% ========================= ONE-PANEL FIGURE =========================
fontName = 'Arial';
fontSize = 16;
lw = 2.8;

x = RS_list(:);

fig = figure('Units','pixels','Position',[80 120 1380 560]);
set(fig,'Renderer','painters');
set(fig,'Color','w');

hold on
box on

colCr        = [0.00 0.00 0.00];   % black
% Stronger colored background lines for all 15 pairwise raw correlations.
% Each pair gets its own color and its own legend entry.
pairColors = lines(nPairs);

pairNames = cell(nPairs,1);
for pp = 1:nPairs
    pairNames{pp} = sprintf('%s-%s',labels{pairs(pp,1)},labels{pairs(pp,2)});
end

% ---------------- all 15 individual raw-correlation curves in background ----------------
% Each pair is summarized across ensembles the same way as C_r:
% ensemble mean curve plus +/- 1 SD shaded band.
%
% Important: the mean raw-pair curves are NOT hard-thresholded. The p-values
% are used only for visual reliability marking. For each R and pair, a point
% is considered visually reliable only if the raw Pearson correlation is
% significant in at least half of the ensembles.
hRawPairs = gobjects(nPairs,1);
rawSigMajority = RawPair_sigFrac >= 0.50;

for pp = 1:nPairs

    ypair = RawPair_mean(:,pp);
    spair = RawPair_std(:,pp);
    thisColor = pairColors(pp,:);
    faintColor = 0.78*[1 1 1] + 0.22*thisColor;

    fill([x; flipud(x)], [ypair+spair; flipud(ypair-spair)], faintColor, ...
        'FaceAlpha',0.05, ...
        'EdgeColor','none', ...
        'HandleVisibility','off');

    % Plot the full mean raw-correlation curve in faint color so that
    % non-significant correlations remain visible but visually downweighted.
    hRawPairs(pp) = plot(x, ypair, '-', ...
        'Color', faintColor, ...
        'LineWidth', 1.35);

    % Marker fill encodes statistical reliability of the raw Pearson pair
    % correlation at each R. The mean raw-correlation values themselves
    % are not removed or hard-thresholded.
    sigHere    = rawSigMajority(:,pp);
    nonsigHere = ~sigHere & isfinite(ypair);

    % Non-significant raw correlations: hollow colored circles.
    if any(nonsigHere)
        plot(x(nonsigHere), ypair(nonsigHere), 'o', ...
            'Color', thisColor, ...
            'MarkerFaceColor','w', ...
            'MarkerEdgeColor',thisColor, ...
            'MarkerSize',5.4, ...
            'LineWidth',1.25, ...
            'LineStyle','none', ...
            'HandleVisibility','off');
    end

    % Significant raw correlations: filled colored circles.
    if any(sigHere)
        plot(x(sigHere), ypair(sigHere), 'o', ...
            'Color', thisColor, ...
            'MarkerFaceColor',thisColor, ...
            'MarkerEdgeColor',thisColor, ...
            'MarkerSize',5.8, ...
            'LineWidth',1.25, ...
            'LineStyle','none', ...
            'HandleVisibility','off');
    end

end

% ---------------- final cooperation rate ----------------
y = CC_mean(:);
s = CC_std(:);

fill([x; flipud(x)], [y+s; flipud(y-s)], colCr, ...
    'FaceAlpha',0.08, ...
    'EdgeColor','none', ...
    'HandleVisibility','off');

h1 = plot(x, y, '-o', ...
    'Color',colCr, ...
    'LineWidth',lw, ...
    'MarkerFaceColor',colCr, ...
    'MarkerEdgeColor',colCr, ...
    'MarkerSize',6);


xlabel('R','FontSize',15,'FontWeight','bold');
ylabel('Value','FontSize',15,'FontWeight','bold');


ax = gca;
set(ax, ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'LineWidth',1.3, ...
    'TickDir','out', ...
    'Color','w', ...
    'XGrid','off', ...
    'YGrid','off', ...
    'XMinorGrid','off', ...
    'YMinorGrid','off');

grid off

xlim([min(RS_list) max(RS_list)]);

% Signed correlations and signed CS can be negative, while C_r is 0 to 1.
ylim([-0.3 1]);

% ---------------- legend and explanatory note ----------------
% We use manual annotation objects instead of a MATLAB legend so the C_r
% legend remains visible even after adding the right-side pair-label panel.

% ---------------- pair legend on right side ----------------
% MATLAB can keep only one normal legend per axes. To guarantee that the
% pair legend appears on the RIGHT side of the output figure, we draw a
% separate legend-like panel using annotation objects in normalized figure
% coordinates. This does not change the plotted data.

% Make room on the right side for the pair legend panel.
ax.Units = 'normalized';
ax.Position = [0.075 0.135 0.705 0.735];

% Right-side pair legend coordinates.
xLine1 = 0.815;
xLine2 = 0.850;
xText  = 0.858;
yTop   = 0.835;
dy     = 0.044;

for pp = 1:nPairs

    yLeg = yTop - (pp-1)*dy;

    annotation(fig,'line',[xLine1 xLine2],[yLeg yLeg], ...
        'Color',pairColors(pp,:), ...
        'LineWidth',2.4);

    annotation(fig,'textbox',[xText yLeg-0.018 0.11 0.035], ...
        'String',pairNames{pp}, ...
        'Interpreter','tex', ...
        'FontName',fontName, ...
        'FontSize',18, ...
        'FontWeight','bold', ...
        'EdgeColor','none', ...
        'BackgroundColor','none', ...
        'VerticalAlignment','middle', ...
        'FitBoxToText','off');

end

% Optional label above the right-side pair legend.
annotation(fig,'textbox',[0.805 0.875 0.16 0.04], ...
    'String','Pair correlations', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',17, ...
    'FontWeight','bold', ...
    'EdgeColor','none', ...
    'BackgroundColor','none', ...
    'HorizontalAlignment','left');


% Manual C_r legend inside the figure.
% This keeps only the C_r curve as a true legend item.
annotation(fig,'line',[0.105 0.145],[0.825 0.825], ...
    'Color',colCr, ...
    'LineWidth',lw);

annotation(fig,'textbox',[0.150 0.805 0.27 0.045], ...
    'String','Final cooperation rate, C_r', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',16, ...
    'FontWeight','bold', ...
    'EdgeColor','white', ...
    'BackgroundColor','white', ...
    'FitBoxToText','off');

% Text-only explanation for the colored curves.
% No line/icon is used here, by design.
annotation(fig,'textbox',[0.105 0.755 0.34 0.055], ...
    'String','Colored curves: raw Pearson correlations', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',15, ...
    'FontWeight','bold', ...
    'EdgeColor','white', ...
    'BackgroundColor','white', ...
    'FitBoxToText','off');

hold off

% Optional export
% exportgraphics(fig,'Supplementary_ordinary_raw_correlations.pdf','ContentType','vector');
% exportgraphics(fig,'Supplementary_ordinary_raw_correlations.png','Resolution',300);

toc

% =============================== FUNCTIONS ===============================
function localIncrement(hWait)

    if isempty(hWait) || ~ishandle(hWait)
        return;
    end

    try

        count = getappdata(hWait,'count');
        total = getappdata(hWait,'total');

        count = count + 1;
        setappdata(hWait,'count',count);

        frac = min(1,count/total);

        if isvalid(hWait)
            waitbar(frac,hWait,sprintf('Running simulations (%.1f%%)',100*frac));
        end

        drawnow limitrate;

    catch
    end

end

function theta = AnglePeriodic_torus(x_to,y_to,x_from,y_from,L)

    dx = x_to - x_from;
    dy = y_to - y_from;

    dx = dx - L*floor(dx/L + 0.5);
    dy = dy - L*floor(dy/L + 0.5);

    theta = atan2(dy,dx);

end

function [A,B] = Finddistance_torus(x,y,r,L)

    x = x(:)';
    y = y(:)';

    N = numel(x);

    DX = x - x.';
    DY = y - y.';

    DX = DX - L*round(DX./L);
    DY = DY - L*round(DY./L);

    D = hypot(DX,DY);
    D(1:N+1:end) = inf;

    [A,B] = find((D > 0) & (D < r));

end

function aa = UpdateThreshold(Used,pi0,Decision0,Pay,Paybefore,ChangeThreshold,noiseInt)

    if Used == 1

        if Decision0 == 1
            dp = -ChangeThreshold;
        else
            dp = ChangeThreshold;
        end

        if Pay ~= 0 && Paybefore ~= 0
            DeltaPay = Pay - Paybefore;
        else
            DeltaPay = 0;
        end

        DeltaPay = DeltaPay + noiseInt*randn;

        aa = pi0 + dp*DeltaPay;

    else

        aa = pi0 + noiseInt*randn;

    end

    aa = min(max(aa,0),1);

end
