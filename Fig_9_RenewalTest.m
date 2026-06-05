% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figure 9: renewal aging test for stripe-crossing events.
%
% What this script does:
%   - Analyzes stripe-crossing inter-event intervals from the representative I1 threshold.
%   - Compares survival, aged-survival, and shuffled-aged survival functions for R = 0.25 and R = 0.45.
%   - Rows are sensing-radius conditions; columns are stripe sizes 0.1, 0.01, and 0.001.
%   - This is a supporting renewal analysis, not a definition of CS.
%   - Randomness: rng('shuffle') is used, so repeated runs are independent.
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

%% ====================== USER SETTINGS ======================
Time        = 1e6;
L           = 2;
dt          = 1;
deltaS      = 0.1;
Noise       = 1e-3;
ThetaDecept = pi/6;
tc          = 3;

RS_list         = [0.25, 0.45];
StripeSize_list = [0.1, 0.01, 0.001];

% Renewal settings
ta = 100;
ST = 2;
EN = 10;

% Windowing settings for Renewal input
DiscardFrac = 0.5;   % remove first half
Slice       = 1e4;   % one window length
WinNumber   = 1;     % choose which window after the cut (1 = first window)

% -------- plotting settings: matched to Figure 10 --------
fontName       = 'Arial';
fontSize       = 12;
labelFontSize  = 18;
legendFontSize = 12;

% Curve colors for clearer separation from the power-law fit
cPsi     = [0.00 0.45 0.74];   % main survival curve
cAge     = [0.85 0.33 0.10];   % aged survival curve
cShAge   = [0.93 0.69 0.13];   % shuffled-aged survival curve
cFit     = [0.55 0.00 0.55];   % dark magenta dashed power-law fit

rng('shuffle');

%% ====================== STORAGE ======================
OUTcell = cell(2,3);   % rows = RS, cols = stripe sizes

%% ====================== MAIN LOOP OVER RS ======================
for ir = 1:numel(RS_list)

    RS = RS_list(ir);

    % S/F params
    rS   = RS * L;
    velS = 0.1;
    rF   = 1 * rS;
    velF = 2 * velS;
    rG   = 1 * rS;

    % Initial conditions
    thetaF          = 2*pi*(rand(1,1) - 0.5);
    thetaFoeF       = zeros(1,1);
    thetaFoeFDECEPT = zeros(1,1);
    thetaS          = 2*pi*(rand(1,2) - 0.5);
    thetaFS         = 2*pi*(rand(1,3) - 0.5);
    thetaFoeSShared = zeros(1,2);

    PayS0 = zeros(2,1);
    PayS  = zeros(2,1);

    ShareInfoS1 = ones(Time,1);
    ShareInfoS2 = ones(Time,1);
    TrustS1     = ones(Time,1);
    TrustS2     = ones(Time,1);
    SharePayS1  = ones(Time,1);
    SharePayS2  = ones(Time,1);

    xF  = L*rand(1,1);   yF  = L*rand(1,1);
    xS  = L*rand(1,2);   yS  = L*rand(1,2);
    xFS = zeros(1,3);    yFS = zeros(1,3);

    Ratio_CC = zeros(Time,1);
    CC = 0;

    %% ---------------------- time loop ----------------------
    for ti = 2:Time

        xFS(1) = xF(1);     yFS(1) = yF(1);
        xFS(2) = xS(1,1);   yFS(2) = yS(1,1);
        xFS(3) = xS(1,2);   yFS(3) = yS(1,2);

        thetaFS(1) = thetaF(1);
        thetaFS(2) = thetaS(1);
        thetaFS(3) = thetaS(2);

        % -------- F angles --------
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rF, L);
        list = l1FS(l2FS == 1);
        list = list(list > 1);

        if ~isempty(list)
            xSthatFpredicts = mean(xFS(list)) + velS*mean(cos(thetaFS(list)))*dt;
            ySthatFpredicts = mean(yFS(list)) + velS*mean(sin(thetaFS(list)))*dt;
            tet1            = AnglePeriodic_torus(xSthatFpredicts, ySthatFpredicts, xFS(1), yFS(1), L);
            thetaDeflect    = ThetaDecept * sign(-1 + 2*rand);

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

        % -------- S angles --------
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

        % -------- decisions --------
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

        % -------- integrate + wrap --------
        xF = mod(xF + velF*cos(thetaF)*dt, L);
        yF = mod(yF + velF*sin(thetaF)*dt, L);
        xS = mod(xS + velS*cos(thetaS)*dt, L);
        yS = mod(yS + velS*sin(thetaS)*dt, L);

        % -------- payoff neighborhood --------
        xFS = [xF(1), xS(1,1), xS(1,2)];
        yFS = [yF(1), yS(1,1), yS(1,2)];
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rG, L);

        list = l1FS(l2FS == 2);  listofF = list(list <= 1);  LF1 = length(listofF);
        list = l1FS(l2FS == 3);  listofF = list(list <= 1);  LF2 = length(listofF);

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

        ShareInfoS1(ti + 1) = UpdateThreshold(1, ShareInfoS1(ti), ShaInfo1,  PayS(1), PayS0(1), deltaS, Noise);
        ShareInfoS2(ti + 1) = UpdateThreshold(1, ShareInfoS2(ti), ShaInfo2,  PayS(2), PayS0(2), deltaS, Noise);
        TrustS1(ti + 1)     = UpdateThreshold(1, TrustS1(ti),     Trusted1,  PayS(1), PayS0(1), deltaS, Noise);
        TrustS2(ti + 1)     = UpdateThreshold(1, TrustS2(ti),     Trusted2,  PayS(2), PayS0(2), deltaS, Noise);
        SharePayS1(ti + 1)  = UpdateThreshold(ShareUSedS1, SharePayS1(ti), s1, PayS(1), PayS0(1), deltaS, Noise);
        SharePayS2(ti + 1)  = UpdateThreshold(ShareUSedS2, SharePayS2(ti), s2, PayS(2), PayS0(2), deltaS, Noise);

        PayS0 = PayS;
        Ratio_CC(ti) = CC / ti;
    end

    %% ---------------------- CUT FIRST HALF, PICK ONE WINDOW FROM I1 ----------------------
    cut0 = floor(DiscardFrac * Time) + 1;
    I1_cut = ShareInfoS1(cut0:end);

    if Slice > numel(I1_cut)
        error('Slice is larger than the remaining I1 after discarding the first half.');
    end

    startIdx = 1 + (WinNumber-1)*Slice;
    endIdx   = startIdx + Slice - 1;

    if endIdx > numel(I1_cut)
        error('Requested window exceeds available post-cut data. Reduce WinNumber or Slice.');
    end

    I1_win = I1_cut(startIdx:endIdx);

    %% ---------------------- Renewal analyses on ONE WINDOW of I1 ----------------------
    for ic = 1:numel(StripeSize_list)
        StripeSize = StripeSize_list(ic);
        outI1 = Renewal(I1_win, StripeSize, ta, ST, EN);
        if isgraphics(gcf), close(gcf); end
        OUTcell{ir,ic} = outI1;
    end
end

%% ====================== PLOTTING ======================
% Formatting is matched to Figure 10:
% - 1200 x 700 pixel figure
% - tiledlayout position [0.08 0.08 0.83 0.84]
% - attached panels: TileSpacing = none, Padding = none
% - Arial font, axis tick font size = 12
% - shared x/y labels font size = 14, bold
% - panel text font size = 12, bold
% - no grid/background guide lines
% - tick-label cleanup identical to Figure 10

fig = figure('Units','pixels','Position',[80 100 1200 700],'Color','w');
tl = tiledlayout(2,3);
tl.TileSpacing = 'none';
tl.Padding     = 'none';
tl.Position    = [0.08 0.08 0.83 0.84];

axs = gobjects(2,3);

for ir = 1:2
    for ic = 1:3

        ax = nexttile;
        axs(ir,ic) = ax;
        hold(ax,'on'); box(ax,'on');

        set(ax,'FontName',fontName,'FontSize',fontSize, ...
            'FontWeight','normal', ...
            'Units','normalized', ...
            'PositionConstraint','outerposition', ...
            'ActivePositionProperty','outerposition', ...
            'LooseInset',[0 0 0 0], ...
            'XScale','log', ...
            'YScale','log', ...
            'TickLabelInterpreter','tex', ...
            'LineWidth',1.1, ...
            'TickDir','out', ...
            'TickLength',[0.018 0.018], ...
            'XMinorTick','off', ...
            'YMinorTick','off', ...
            'Layer','top');

        OUT = OUTcell{ir,ic};

        h1 = loglog(ax, OUT.tau_axis, OUT.Psi, ...
            'Color',cPsi, ...
            'LineWidth',2.5, ...
            'DisplayName','$\Psi(\tau)$');
        h2 = loglog(ax, OUT.tau_axis, OUT.PsiAge, ...
            'Color',cAge, ...
            'LineWidth',2.5, ...
            'DisplayName','Aged $\Psi(\tau)$');
        h3 = loglog(ax, OUT.tau_axis, OUT.PsiSHAge, ...
            'Color',cShAge, ...
            'LineWidth',2.5, ...
            'DisplayName','Shuffled-Aged $\Psi(\tau)$');

        % Plot the power-law fit last, thicker, and in dark magenta so it is visible
        % even where it overlaps the survival curve.
        h4 = loglog(ax, OUT.tau_fit, OUT.Psi_fit, ...
            'LineStyle','--', ...
            'Color',cFit, ...
            'LineWidth',3.2, ...
            'DisplayName','Power-law fit');
        uistack(h4,'top');

        ylim(ax,[1e-3 1]);

        ylabel(ax,'');

        if ir == 2 && ic == 2
            xlabel(ax,'$\tau$', ...
                'Interpreter','latex', ...
                'FontName',fontName, ...
                'FontSize',labelFontSize, ...
                'FontWeight','bold');
        else
            xlabel(ax,'');
        end

        if ir < 2
            set(ax,'XTickLabel',[]);
        end
        if ic > 1
            set(ax,'YTickLabel',[]);
        end

        text(ax,0.055,0.08, ...
            { ['R = ' num2str(RS_list(ir),'%.2f')] ; ...
              ['Stripe = ' num2str(StripeSize_list(ic),'%.3g')] ; ...
              ['\mu - 1 = ' num2str(OUT.mu_minus_1,'%.3f')] }, ...
            'Units','normalized', ...
            'VerticalAlignment','bottom', ...
            'Interpreter','tex', ...
            'FontName',fontName, ...
            'FontSize',legendFontSize, ...
            'FontWeight','bold', ...
            'BackgroundColor','w', ...
            'EdgeColor','none', ...
            'Margin',2);

        % Only one legend for common line meanings, inside the top-right panel.
        if ir == 1 && ic == 3
            lgd = legend(ax,[h1 h2 h3 h4], ...
                {'$\Psi(\tau)$','Aged $\Psi(\tau)$','Shuffled-Aged $\Psi(\tau)$','Power-law fit'}, ...
                'Location','northeast');
            lgd.FontName    = fontName;
            lgd.FontSize    = legendFontSize;
            lgd.FontWeight  = 'bold';
            lgd.Interpreter = 'latex';
            lgd.Box         = 'on';
        end

        hold(ax,'off');
    end
end

drawnow;
for r = 1:2
    for c = 1:3
        ax = axs(r,c);
        ax.Position   = ax.OuterPosition;
        ax.LooseInset = [0 0 0 0];
    end
end
drawnow;

%% -------- final clean removal of touching border tick labels --------
for r = 1:2
    for c = 1:3
        ax = axs(r,c);

        if c == 1
            ylab = string(ax.YTickLabel);
            if ~isempty(ylab)
                if r < 2 && numel(ylab) >= 1
                    ylab(1) = "";
                end
                if r > 1 && numel(ylab) >= 1
                    ylab(end) = "";
                end
                ax.YTickLabel = ylab;
            end
        end

        if r == 2
            xlab = string(ax.XTickLabel);
            if ~isempty(xlab)
                if c == 1
                    xlab(end) = "";
                elseif c == 2
                    xlab(1)   = "";
                    xlab(end) = "";
                elseif c == 3
                    xlab(1) = "";
                end
                ax.XTickLabel = xlab;
            end
        end
    end
end

%% -------- shared y-label centered between the two left panels --------
drawnow
posTopLeft = axs(1,1).Position;
posBotLeft = axs(2,1).Position;

left_edge_leftcol   = min(posTopLeft(1), posBotLeft(1));
bottom_edge_leftcol = min(posTopLeft(2), posBotLeft(2));
top_edge_leftcol    = max(posTopLeft(2)+posTopLeft(4), posBotLeft(2)+posBotLeft(4));

y_center = 0.5 * (bottom_edge_leftcol + top_edge_leftcol);
x_text   = left_edge_leftcol - 0.045;

x_text   = max(0.015, min(x_text, 0.98));
y_center = max(0.05, min(y_center, 0.95));

han = axes(fig,'Position',[0 0 1 1],'Visible','off','Units','normalized');
han.HitTest = 'off';
text(han, x_text, y_center, '$\Psi(\tau)$', ...
    'Interpreter','latex', ...
    'Rotation',90, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'FontName',fontName, ...
    'FontSize',labelFontSize, ...
    'FontWeight','bold', ...
    'Units','normalized');

uistack(han,'bottom');

drawnow;

toc

%% =============================== LOCAL FUNCTIONS ===============================
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

    DeltaPay = DeltaPay + noiseInt*randn;
    aa = pi0 + dp * DeltaPay;
else
    aa = pi0 + noiseInt*randn;
end

aa = min(max(aa,0),1);
end

function trimSharedTickLabels(ax,whichAxis,idx,nTot)
switch lower(whichAxis)
    case 'x'
        labs = string(ax.XTickLabel);
        if isempty(labs), return; end

        if idx > 1 && numel(labs) >= 1
            labs(1) = "";
        end
        if idx < nTot && numel(labs) >= 1
            labs(end) = "";
        end

        ax.XTickLabel = labs;

    case 'y'
        labs = string(ax.YTickLabel);
        if isempty(labs), return; end

        if idx > 1 && numel(labs) >= 1
            labs(end) = "";
        end
        if idx < nTot && numel(labs) >= 1
            labs(1) = "";
        end

        ax.YTickLabel = labs;
end
end
