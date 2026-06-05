% Korosh Mahmoodi
%
% Paper: Complexity synchronization as a diagnostic and control principle
%        for adaptive systems
%
% Purpose: Figures 13-17: CS-guided rescue and supplementary rescue analyses.
%
% What this script does:
%   - Runs perturbation and rescue simulations for baseline, damage, random/local rescue, and CS-guided targeted rescue.
%   - Generates MDEA CS networks, cooperation recovery curves, search-efficiency curves, and optional low-R DFA supplementary results.
%   - Targeted rescue restores the damaged payoff-sharing pair P1-P2; random/local rescue modifies a non-targeted pair.
%   - Default Time is set for quicker testing in the uploaded version; set Time = 1e6 for manuscript-scale runs.
%   - Randomness: by default a fresh base seed is generated at each run; set useFreshBaseSeed = false for exact reproducibility.
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
cleanup_all_progress_bars();
progressBarCleanupObj = onCleanup(@() cleanup_all_progress_bars());

%% ===================== USER PARAMETERS =====================
Time   = 1e5;      % quick-test default; set to 1e6 for manuscript-scale runs
L      = 2;
dt     = 1;
RSfac  = 0.45;      % main rescue demonstration: high-cooperation/event-driven regime
RSfac_lowDFA = 0.25; % supplementary robustness: low-coordination/persistence-sensitive regime
makeLowRS_DFA_supplement = true;
Noise  = 1e-3;
str    = 1e-3;
EnvSuccessProb = 1;

% Learning-step levels
highDelta = 0.100;
lowDelta  = 0.025;

% Channel order: [I1 T1 P1 I2 T2 P2]
labels      = {'I1','T1','P1','I2','T2','P2'};
targetPair  = [3 6];   % P1-P2, CS-guided target
randomPairA = [1 5];   % I1-T2, wrong local rescue control; change if desired

% Colorbar settings
MDEA_wMin = 0.5;
MDEA_wMax = 1.0;
MDEA_cbTicks = [0.5 0.6 0.7 0.8 0.9 1.0];
DFA_wMin = 0.0;
DFA_wMax = 1.0;
DFA_cbTicks = [0 0.25 0.5 0.75 1.0];

% MDEA settings
fit_ST = 0.1;
fit_EN = 0.9;
Slice   = 1e4;
Overlap = floor(0.75 * Slice);
Newdata = Slice - Overlap;

% DFA settings for optional low-R supplementary diagnosis
dfa_pts   = 100:100:1000;
dfa_order = 1;

% Random-seed architecture
% IMPORTANT:
%   useFreshBaseSeed = true  -> every time you press Run, a new base seed is created.
%   Every condition, replicate, rescue strength, and tested pair then receives a
%   distinct derived seed. Therefore no two independent simulations reuse the
%   same random-number stream, and rerunning the script produces different results.
%
%   For exact debugging/reproducibility only, set useFreshBaseSeed = false and
%   fixedBaseSeed = 12345. For final stochastic exploration, keep true.
useFreshBaseSeed = true;
fixedBaseSeed    = 12345;

if useFreshBaseSeed
    rng('shuffle');
    baseSeed = randi(2^31 - 1);
else
    baseSeed = fixedBaseSeed;
end

fprintf('\nRandom seed architecture: baseSeed = %d\n', baseSeed);

% Ensemble controls
ENS = 2;              % quick-test default; increase to 10 for manuscript-scale averaging

% Figure B/C controls
sList = 0:0.1:1;
nRepB = ENS;           % keep rescue curves consistent with the rest of the paper
nRandomOrders = 1000;  % random-search permutations for Figure C

% Parallel control
useParfor = true;
numWorkers = [];       % [] = MATLAB decides; or set e.g., 8

% Visible progress bars. For PARFOR sections the bar is updated from workers
% through a DataQueue. This keeps the run responsive without changing simulation logic.
showProgressBars = true;

if useParfor
    try
        poolObj = gcp('nocreate');
        if isempty(poolObj)
            if isempty(numWorkers)
                parpool;
            else
                parpool(numWorkers);
            end
        end
    catch ME
        warning('Could not start/use parallel pool. Falling back to regular FOR. Message: %s', ME.message);
        useParfor = false;
    end
end

% Output
outputFolder = 'C:\Users\ARL\Downloads';
if ~exist(outputFolder, 'dir')
    outputFolder = pwd;
end
pdfFile = fullfile(outputFolder, 'CS_Guided_Rescue_Figures_A_B_C_PARFOR.pdf');
matFile = fullfile(outputFolder, 'CS_Guided_Rescue_Figures_A_B_C_PARFOR_results.mat');
if exist(pdfFile, 'file'), delete(pdfFile); end

%% ===================== DELTA VECTORS =====================
Delta_baseline = highDelta * ones(1,6);

Delta_damage = highDelta * ones(1,6);
Delta_damage(targetPair) = lowDelta;

% Figure A random/local rescue: same local-control idea, but on a wrong pair.
% P1-P2 remains damaged; the wrong pair is boosted to test non-guided repair.
Delta_randomA = Delta_damage;
Delta_randomA(randomPairA) = highDelta + (highDelta - lowDelta);

% Figure A targeted rescue: restore only the diagnosed P1-P2 subsystem.
Delta_targetA = Delta_damage;
Delta_targetA(targetPair) = highDelta;

condNames_A = {'Baseline','Damage','Random/local rescue','CS-guided targeted rescue'};
Delta_A = [Delta_baseline; Delta_damage; Delta_randomA; Delta_targetA];
nCondA = size(Delta_A,1);

%% ===================== FIGURE A: CS NETWORKS =====================
fprintf('\nFigure A: running %d MDEA network conditions x ENS = %d...\n', nCondA, ENS);

% Ensemble-averaged network: each condition is run ENS times with distinct seeds.
% We average the CS matrices and Cr values across ensembles. This matches the
% rest of the paper where ENS = 10 independent stochastic simulations are used.
M_A_ens  = nan(6,6,nCondA,ENS);
Cr_A_ens = nan(nCondA,ENS);
Pr_A_ens = nan(nCondA,ENS);
seed_A = nan(nCondA,ENS);
for c = 1:nCondA
    for ee = 1:ENS
        seed_A(c,ee) = baseSeed + 10000*c + 100*ee + 17;
    end
end

nTasksA = nCondA * ENS;
M_A_task  = cell(1,nTasksA);
Cr_A_task = nan(1,nTasksA);
Pr_A_task = nan(1,nTasksA);
cond_A_task = nan(1,nTasksA);
ens_A_task  = nan(1,nTasksA);
seed_A_task = nan(1,nTasksA);
kkA = 0;
for c = 1:nCondA
    for ee = 1:ENS
        kkA = kkA + 1;
        cond_A_task(kkA) = c;
        ens_A_task(kkA)  = ee;
        seed_A_task(kkA) = seed_A(c,ee);
    end
end

hProgA = init_progress_bar(showProgressBars, 'Figure A: MDEA networks', nTasksA);
progressQueueA = [];
if useParfor && showProgressBars
    progressQueueA = parallel.pool.DataQueue;
    afterEach(progressQueueA, @(~) update_progress_bar(hProgA, nTasksA, 'Figure A: MDEA networks'));
end

if useParfor
    parfor tt = 1:nTasksA
        c = cond_A_task(tt);
        seedNow = seed_A_task(tt); % distinct seed per condition and ensemble
        [signals, Ratio_Cr, Ratio_Pr] = run_single_condition_deltaVec( ...
            Time, L, dt, RSfac, Noise, EnvSuccessProb, Delta_A(c,:), seedNow, false);

        Cr_A_task(tt) = Ratio_Cr(end);
        Pr_A_task(tt) = Ratio_Pr(end);

        [Mtmp, ~] = compute_cs_matrix_mdea(signals, Slice, Newdata, str, fit_ST, fit_EN);
        M_A_task{tt} = Mtmp;
        if showProgressBars
            send(progressQueueA, tt);
        end
    end
else
    for tt = 1:nTasksA
        c  = cond_A_task(tt);
        ee = ens_A_task(tt);
        fprintf('  Figure A condition %d/%d (%s), ensemble %d/%d\n', c, nCondA, condNames_A{c}, ee, ENS);
        seedNow = seed_A_task(tt);
        [signals, Ratio_Cr, Ratio_Pr] = run_single_condition_deltaVec( ...
            Time, L, dt, RSfac, Noise, EnvSuccessProb, Delta_A(c,:), seedNow, false);

        Cr_A_task(tt) = Ratio_Cr(end);
        Pr_A_task(tt) = Ratio_Pr(end);

        [Mtmp, ~] = compute_cs_matrix_mdea(signals, Slice, Newdata, str, fit_ST, fit_EN);
        M_A_task{tt} = Mtmp;
        update_progress_bar(hProgA, nTasksA, 'Figure A: MDEA networks');
    end
end
close_progress_bar(hProgA);

for tt = 1:nTasksA
    c  = cond_A_task(tt);
    ee = ens_A_task(tt);
    M_A_ens(:,:,c,ee) = M_A_task{tt};
    Cr_A_ens(c,ee)    = Cr_A_task(tt);
    Pr_A_ens(c,ee)    = Pr_A_task(tt);
end

M_A      = mean(M_A_ens,4,'omitnan');
Cr_A     = mean(Cr_A_ens,2,'omitnan')';
Cr_A_sd  = std(Cr_A_ens,0,2,'omitnan')';
Pr_A     = mean(Pr_A_ens,2,'omitnan')';

figA = figure('Color','w','Units','pixels','Position',[40 40 1300 1050]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
for c = 1:nCondA
    nexttile;
    ttl = sprintf('%s\nC_r = %.3f ± %.3f', condNames_A{c}, Cr_A(c), Cr_A_sd(c));
    plot_cs_network(M_A(:,:,c), labels, ttl, MDEA_wMin, MDEA_wMax, MDEA_cbTicks);
end
sgtitle(sprintf('CS-guided intervention network test (MDEA, R = %.2f, ENS = %d)', RSfac, ENS), ...
    'FontSize',18,'FontWeight','bold');
set(figA,'PaperPositionMode','auto');
exportgraphics(figA, pdfFile, 'ContentType','vector', 'Append', false);

%% ===================== FIGURE B: COOPERATION RECOVERY CURVES =====================
fprintf('\nFigure B: running rescue curves with FAST Cr-only PARFOR simulations...\n');

% Baseline and damage replicates
Cr_base_rep = nan(1,nRepB);
Cr_dmg_rep  = nan(1,nRepB);
seed_B_base = baseSeed + 100000 + 1000*(1:nRepB) + 1;
seed_B_dmg  = baseSeed + 100000 + 1000*(1:nRepB) + 2;

hProgB0 = init_progress_bar(showProgressBars, 'Figure B: baseline/damage replicates', nRepB);
progressQueueB0 = [];
if useParfor && showProgressBars
    progressQueueB0 = parallel.pool.DataQueue;
    afterEach(progressQueueB0, @(~) update_progress_bar(hProgB0, nRepB, 'Figure B: baseline/damage replicates'));
end

if useParfor
    parfor rr = 1:nRepB
        seedBase = seed_B_base(rr);
        seedDmg  = seed_B_dmg(rr);
        Cr_base_rep(rr) = run_single_condition_Cr_only(Time,L,dt,RSfac,Noise,EnvSuccessProb,Delta_baseline,seedBase);
        Cr_dmg_rep(rr)  = run_single_condition_Cr_only(Time,L,dt,RSfac,Noise,EnvSuccessProb,Delta_damage,seedDmg);
        if showProgressBars
            send(progressQueueB0, rr);
        end
    end
else
    for rr = 1:nRepB
        fprintf('  Figure B baseline/damage replicate %d/%d\n', rr, nRepB);
        seedBase = seed_B_base(rr);
        seedDmg  = seed_B_dmg(rr);
        Cr_base_rep(rr) = run_single_condition_Cr_only(Time,L,dt,RSfac,Noise,EnvSuccessProb,Delta_baseline,seedBase);
        Cr_dmg_rep(rr)  = run_single_condition_Cr_only(Time,L,dt,RSfac,Noise,EnvSuccessProb,Delta_damage,seedDmg);
        update_progress_bar(hProgB0, nRepB, 'Figure B: baseline/damage replicates');
    end
end
close_progress_bar(hProgB0);

% Build task list for random/global/targeted rescue curves
% Columns: [strategy, sIndex, replicate]
% strategy: 1=random/local, 2=equal-budget global, 3=targeted
nS = numel(sList);
taskList = zeros(3*nS*nRepB,3);
kk = 0;
for rr = 1:nRepB
    for ss = 1:nS
        for strategy = 1:3
            kk = kk + 1;
            taskList(kk,:) = [strategy, ss, rr];
        end
    end
end
nTasks = size(taskList,1);
Cr_task = nan(nTasks,1);
seed_task = nan(nTasks,1);
for tt = 1:nTasks
    strategy = taskList(tt,1);
    ss       = taskList(tt,2);
    rr       = taskList(tt,3);
    seed_task(tt) = baseSeed + 200000 + 10000*strategy + 100*ss + rr;
end

hProgB = init_progress_bar(showProgressBars, 'Figure B: rescue curves', nTasks);
progressQueueB = [];
if useParfor && showProgressBars
    progressQueueB = parallel.pool.DataQueue;
    afterEach(progressQueueB, @(~) update_progress_bar(hProgB, nTasks, 'Figure B: rescue curves'));
end

if useParfor
    parfor tt = 1:nTasks
        strategy = taskList(tt,1);
        ss       = taskList(tt,2);
        rr       = taskList(tt,3);
        s        = sList(ss);
        seedNow  = seed_task(tt); % distinct seed per rescue simulation

        Delta_now = make_rescue_delta(strategy, s, Delta_damage, randomPairA, targetPair, highDelta, lowDelta);
        Cr_task(tt) = run_single_condition_Cr_only(Time,L,dt,RSfac,Noise,EnvSuccessProb,Delta_now,seedNow);
        if showProgressBars
            send(progressQueueB, tt);
        end
    end
else
    for tt = 1:nTasks
        strategy = taskList(tt,1);
        ss       = taskList(tt,2);
        rr       = taskList(tt,3);
        s        = sList(ss);
        seedNow  = seed_task(tt); % distinct seed per rescue simulation
        fprintf('  Figure B task %d/%d, strategy=%d, s=%.2f, rep=%d\n', tt, nTasks, strategy, s, rr);

        Delta_now = make_rescue_delta(strategy, s, Delta_damage, randomPairA, targetPair, highDelta, lowDelta);
        Cr_task(tt) = run_single_condition_Cr_only(Time,L,dt,RSfac,Noise,EnvSuccessProb,Delta_now,seedNow);
        update_progress_bar(hProgB, nTasks, 'Figure B: rescue curves');
    end
end
close_progress_bar(hProgB);

Cr_random = nan(nS, nRepB);
Cr_global = nan(nS, nRepB);
Cr_target = nan(nS, nRepB);
for tt = 1:nTasks
    strategy = taskList(tt,1);
    ss       = taskList(tt,2);
    rr       = taskList(tt,3);
    if strategy == 1
        Cr_random(ss,rr) = Cr_task(tt);
    elseif strategy == 2
        Cr_global(ss,rr) = Cr_task(tt);
    else
        Cr_target(ss,rr) = Cr_task(tt);
    end
end

baseMean = mean(Cr_base_rep,'omitnan');
dmgMean  = mean(Cr_dmg_rep,'omitnan');

figB = figure('Color','w','Units','pixels','Position',[120 120 1200 650]); hold on
[hRandB,   ~] = plot_mean_sd_band(sList, Cr_random, 1);
[hGlobB,   ~] = plot_mean_sd_band(sList, Cr_global, 2);
[hTargB,   ~] = plot_mean_sd_band(sList, Cr_target, 3);
hBaseB = yline(baseMean, '--', sprintf('Baseline %.3f', baseMean), ...
    'LineWidth',1.8, 'LabelHorizontalAlignment','left');
hDmgB  = yline(dmgMean,  ':',  sprintf('Damage %.3f', dmgMean), ...
    'LineWidth',1.8, 'LabelHorizontalAlignment','left');
set([hBaseB hDmgB], 'FontSize',16, 'FontWeight','bold', 'FontName','Arial');

xlabel('Rescue strength, s','FontSize',14,'FontWeight','bold');
ylabel('Cooperation rate, C_r','FontSize',14,'FontWeight','bold');
title(sprintf('Cooperation recovery under rescue mechanisms (R = %.2f, ENS = %d)', RSfac, ENS), ...
    'FontSize',15,'FontWeight','bold');
hLegB = legend([hRandB hGlobB hTargB hBaseB hDmgB], ...
    {'Random/local rescue', 'Equal-budget global rescue', ...
     'CS-guided targeted rescue', 'Baseline', 'Damage'}, ...
    'Location','northwest');
grid off; box on
set(gca,'FontName','Arial','FontSize',13,'FontWeight','bold');
exportgraphics(figB, pdfFile, 'ContentType','vector', 'Append', true);

%% ===================== FIGURE C: SEARCH EFFICIENCY =====================
fprintf('\nFigure C: testing all local rescue pairs x ENS = %d with FAST Cr-only PARFOR simulations...\n', ENS);
pairs = nchoosek(1:6,2);
nPairs = size(pairs,1);
Cr_pair_ens = nan(nPairs,ENS);
seed_pair = nan(nPairs,ENS);
for pp = 1:nPairs
    for ee = 1:ENS
        seed_pair(pp,ee) = baseSeed + 300000 + 1000*pp + ee;
    end
end

nPairTasks = nPairs * ENS;
pair_task = nan(nPairTasks,1);
ens_pair_task = nan(nPairTasks,1);
seed_pair_task = nan(nPairTasks,1);
kkP = 0;
for pp = 1:nPairs
    for ee = 1:ENS
        kkP = kkP + 1;
        pair_task(kkP) = pp;
        ens_pair_task(kkP) = ee;
        seed_pair_task(kkP) = seed_pair(pp,ee);
    end
end
Cr_pair_task = nan(nPairTasks,1);

hProgC = init_progress_bar(showProgressBars, 'Figure C: pair search', nPairTasks);
progressQueueC = [];
if useParfor && showProgressBars
    progressQueueC = parallel.pool.DataQueue;
    afterEach(progressQueueC, @(~) update_progress_bar(hProgC, nPairTasks, 'Figure C: pair search'));
end

if useParfor
    parfor tt = 1:nPairTasks
        pp = pair_task(tt);
        Delta_pair = Delta_damage;
        Delta_pair(pairs(pp,:)) = highDelta;
        seedNow = seed_pair_task(tt); % distinct seed per pair and ensemble
        Cr_pair_task(tt) = run_single_condition_Cr_only(Time,L,dt,RSfac,Noise,EnvSuccessProb,Delta_pair,seedNow);
        if showProgressBars
            send(progressQueueC, tt);
        end
    end
else
    for tt = 1:nPairTasks
        pp = pair_task(tt);
        ee = ens_pair_task(tt);
        fprintf('  Figure C pair %d/%d: %s-%s, ensemble %d/%d\n', pp, nPairs, labels{pairs(pp,1)}, labels{pairs(pp,2)}, ee, ENS);
        Delta_pair = Delta_damage;
        Delta_pair(pairs(pp,:)) = highDelta;
        seedNow = seed_pair_task(tt);
        Cr_pair_task(tt) = run_single_condition_Cr_only(Time,L,dt,RSfac,Noise,EnvSuccessProb,Delta_pair,seedNow);
        update_progress_bar(hProgC, nPairTasks, 'Figure C: pair search');
    end
end
close_progress_bar(hProgC);

for tt = 1:nPairTasks
    pp = pair_task(tt);
    ee = ens_pair_task(tt);
    Cr_pair_ens(pp,ee) = Cr_pair_task(tt);
end
Cr_pair = mean(Cr_pair_ens,2,'omitnan');
Cr_pair_sd = std(Cr_pair_ens,0,2,'omitnan');

% CS-guided search order: test the diagnosed P1-P2 pair first, then the rest.
targetIdx = find(all(sort(pairs,2) == sort(targetPair),2));
csOrder = [targetIdx; setdiff((1:nPairs)', targetIdx, 'stable')];
csBest = cummax(Cr_pair(csOrder));

% Random-search distribution: random permutations of all pairs.
rng(baseSeed + 888, 'twister'); % fresh each script run because baseSeed is fresh
randBest = nan(nPairs, nRandomOrders);
for rr = 1:nRandomOrders
    ord = randperm(nPairs);
    randBest(:,rr) = cummax(Cr_pair(ord));
end
randMean = mean(randBest,2,'omitnan');
randSD   = std(randBest,0,2,'omitnan');
xSearch = (1:nPairs)';

figC = figure('Color','w','Units','pixels','Position',[140 140 1050 720]);
axC = axes(figC); hold(axC,'on');

% Make the random-search envelope visible and robust in PDF export.
yLower = randMean - randSD;
yUpper = randMean + randSD;
yLower(~isfinite(yLower)) = randMean(~isfinite(yLower));
yUpper(~isfinite(yUpper)) = randMean(~isfinite(yUpper));
hRandBandC = fill(axC, [xSearch; flipud(xSearch)], [yLower; flipud(yUpper)], ...
    [0.80 0.80 0.80], 'EdgeColor','none', 'FaceAlpha',0.33, ...
    'HandleVisibility','off');

hRandC = plot(axC, xSearch, randMean, 'o-', 'LineWidth',2.2, 'MarkerSize',6);
hCSC   = plot(axC, xSearch, csBest, '^-', 'LineWidth',2.8, 'MarkerSize',8);
hBaseC = yline(axC, baseMean, '--', 'Baseline', ...
    'LineWidth',1.9, 'LabelHorizontalAlignment','left');
hDmgC  = yline(axC, dmgMean,  ':',  'Damage', ...
    'LineWidth',1.9, 'LabelHorizontalAlignment','left');
set([hBaseC hDmgC], 'FontSize',16, 'FontWeight','bold', 'FontName','Arial');

xlim(axC, [1 nPairs]);
yMaxC = max([baseMean; dmgMean; randMean(:)+randSD(:); csBest(:)], [], 'omitnan');
yMinC = min([dmgMean; randMean(:)-randSD(:); csBest(:)], [], 'omitnan');
ylim(axC, [max(-0.05,yMinC-0.05), min(0.85,yMaxC+0.08)]);
xticks(axC, 1:2:nPairs);

xlabel(axC, 'Number of tested rescue pairs','FontSize',14,'FontWeight','bold');
ylabel(axC, 'Best achieved cooperation rate, C_r','FontSize',14,'FontWeight','bold');
title(axC, sprintf('CS-guided rescue reduces intervention search (R = %.2f)', RSfac), ...
    'FontSize',15,'FontWeight','bold');
hLegC = legend(axC, [hRandC hCSC hBaseC hDmgC], ...
    {'Random search','CS-guided search','Baseline','Damage'}, ...
    'Location','southeast');
grid(axC,'off'); box(axC,'on');
set(axC,'FontName','Arial','FontSize',13,'FontWeight','bold','Layer','top');
drawnow;

% Export Figure C as an image inside the PDF. This avoids occasional blank/faint
% vector rendering of the shaded envelope in some PDF viewers/parsers.
exportgraphics(figC, pdfFile, 'ContentType','image', 'Resolution',300, 'Append', true);


%% ===================== SUPPLEMENTARY FIGURE D/E: LOW-R DFA ROBUSTNESS =====================
% Rationale:
%   R = 0.45 is the cooperative regime where MDEA-CS is the main diagnostic.
%   R = 0.25 is already weakly coordinated in MDEA, so DFA can be useful as a
%   persistence/memory-sensitive diagnostic. These figures should usually go
%   to Supplementary Information, not the main text.
if makeLowRS_DFA_supplement
    fprintf('\nSupplementary Figure D: running low-R DFA network conditions at R = %.2f x ENS = %d...\n', RSfac_lowDFA, ENS);

    M_D_low_ens = nan(6,6,nCondA,ENS);
    Cr_D_low_ens = nan(nCondA,ENS);
    seed_D_low = nan(nCondA,ENS);
    for c = 1:nCondA
        for ee = 1:ENS
            seed_D_low(c,ee) = baseSeed + 400000 + 10000*c + 100*ee + 31;
        end
    end

    nTasksD = nCondA * ENS;
    M_D_low_task  = cell(1,nTasksD);
    Cr_D_low_task = nan(1,nTasksD);
    cond_D_task   = nan(1,nTasksD);
    ens_D_task    = nan(1,nTasksD);
    seed_D_task   = nan(1,nTasksD);
    kkD = 0;
    for c = 1:nCondA
        for ee = 1:ENS
            kkD = kkD + 1;
            cond_D_task(kkD) = c;
            ens_D_task(kkD)  = ee;
            seed_D_task(kkD) = seed_D_low(c,ee);
        end
    end

    hProgD = init_progress_bar(showProgressBars, 'Supplement D: DFA networks', nTasksD);
    progressQueueD = [];
    if useParfor && showProgressBars
        progressQueueD = parallel.pool.DataQueue;
        afterEach(progressQueueD, @(~) update_progress_bar(hProgD, nTasksD, 'Supplement D: DFA networks'));
    end

    if useParfor
        parfor tt = 1:nTasksD
            c = cond_D_task(tt);
            seedNow = seed_D_task(tt);
            [signals, Ratio_Cr, ~] = run_single_condition_deltaVec( ...
                Time, L, dt, RSfac_lowDFA, Noise, EnvSuccessProb, Delta_A(c,:), seedNow, false);
            Cr_D_low_task(tt) = Ratio_Cr(end);
            [Mtmp, ~] = compute_cs_matrix_dfa(signals, Slice, Newdata, dfa_pts, dfa_order);
            M_D_low_task{tt} = Mtmp;
            if showProgressBars
                send(progressQueueD, tt);
            end
        end
    else
        for tt = 1:nTasksD
            c = cond_D_task(tt);
            ee = ens_D_task(tt);
            fprintf('  Low-R DFA condition %d/%d (%s), ensemble %d/%d\n', c, nCondA, condNames_A{c}, ee, ENS);
            seedNow = seed_D_task(tt);
            [signals, Ratio_Cr, ~] = run_single_condition_deltaVec( ...
                Time, L, dt, RSfac_lowDFA, Noise, EnvSuccessProb, Delta_A(c,:), seedNow, false);
            Cr_D_low_task(tt) = Ratio_Cr(end);
            [Mtmp, ~] = compute_cs_matrix_dfa(signals, Slice, Newdata, dfa_pts, dfa_order);
            M_D_low_task{tt} = Mtmp;
            update_progress_bar(hProgD, nTasksD, 'Supplement D: DFA networks');
        end
    end
    close_progress_bar(hProgD);

    for tt = 1:nTasksD
        c = cond_D_task(tt);
        ee = ens_D_task(tt);
        M_D_low_ens(:,:,c,ee) = M_D_low_task{tt};
        Cr_D_low_ens(c,ee)    = Cr_D_low_task(tt);
    end

    M_D_low     = mean(M_D_low_ens,4,'omitnan');
    Cr_D_low    = mean(Cr_D_low_ens,2,'omitnan')';
    Cr_D_low_sd = std(Cr_D_low_ens,0,2,'omitnan')';

    figD = figure('Color','w','Units','pixels','Position',[60 60 1300 1050]);
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    for c = 1:nCondA
        nexttile;
        ttl = sprintf('%s\nC_r = %.3f ± %.3f', condNames_A{c}, Cr_D_low(c), Cr_D_low_sd(c));
        plot_cs_network(M_D_low(:,:,c), labels, ttl, DFA_wMin, DFA_wMax, DFA_cbTicks);
    end
    sgtitle(sprintf('DFA rescue-network diagnostic (R = %.2f, ENS = %d)', RSfac_lowDFA, ENS), ...
        'FontSize',18,'FontWeight','bold');
    set(figD,'PaperPositionMode','auto');
    exportgraphics(figD, pdfFile, 'ContentType','vector', 'Append', true);

    %% ----- Supplementary Figure E: low-R Cr-only recovery curves -----
    fprintf('\nSupplementary Figure E: running low-R rescue curves with FAST Cr-only simulations...\n');

    Cr_base_low_rep = nan(1,nRepB);
    Cr_dmg_low_rep  = nan(1,nRepB);
    seed_E_base = baseSeed + 500000 + 1000*(1:nRepB) + 1;
    seed_E_dmg  = baseSeed + 500000 + 1000*(1:nRepB) + 2;

    hProgE0 = init_progress_bar(showProgressBars, 'Supplement E: baseline/damage replicates', nRepB);
    progressQueueE0 = [];
    if useParfor && showProgressBars
        progressQueueE0 = parallel.pool.DataQueue;
        afterEach(progressQueueE0, @(~) update_progress_bar(hProgE0, nRepB, 'Supplement E: baseline/damage replicates'));
    end

    if useParfor
        parfor rr = 1:nRepB
            Cr_base_low_rep(rr) = run_single_condition_Cr_only(Time,L,dt,RSfac_lowDFA,Noise,EnvSuccessProb,Delta_baseline,seed_E_base(rr));
            Cr_dmg_low_rep(rr)  = run_single_condition_Cr_only(Time,L,dt,RSfac_lowDFA,Noise,EnvSuccessProb,Delta_damage,seed_E_dmg(rr));
            if showProgressBars
                send(progressQueueE0, rr);
            end
        end
    else
        for rr = 1:nRepB
            Cr_base_low_rep(rr) = run_single_condition_Cr_only(Time,L,dt,RSfac_lowDFA,Noise,EnvSuccessProb,Delta_baseline,seed_E_base(rr));
            Cr_dmg_low_rep(rr)  = run_single_condition_Cr_only(Time,L,dt,RSfac_lowDFA,Noise,EnvSuccessProb,Delta_damage,seed_E_dmg(rr));
            update_progress_bar(hProgE0, nRepB, 'Supplement E: baseline/damage replicates');
        end
    end
    close_progress_bar(hProgE0);

    taskListLow = taskList;
    nTasksLow = size(taskListLow,1);
    Cr_task_low = nan(nTasksLow,1);
    seed_task_low = nan(nTasksLow,1);
    for tt = 1:nTasksLow
        strategy = taskListLow(tt,1);
        ss       = taskListLow(tt,2);
        rr       = taskListLow(tt,3);
        seed_task_low(tt) = baseSeed + 600000 + 10000*strategy + 100*ss + rr;
    end

    hProgE = init_progress_bar(showProgressBars, 'Supplement E: rescue curves', nTasksLow);
    progressQueueE = [];
    if useParfor && showProgressBars
        progressQueueE = parallel.pool.DataQueue;
        afterEach(progressQueueE, @(~) update_progress_bar(hProgE, nTasksLow, 'Supplement E: rescue curves'));
    end

    if useParfor
        parfor tt = 1:nTasksLow
            strategy = taskListLow(tt,1);
            ss       = taskListLow(tt,2);
            s        = sList(ss);
            Delta_now = make_rescue_delta(strategy, s, Delta_damage, randomPairA, targetPair, highDelta, lowDelta);
            Cr_task_low(tt) = run_single_condition_Cr_only(Time,L,dt,RSfac_lowDFA,Noise,EnvSuccessProb,Delta_now,seed_task_low(tt));
            if showProgressBars
                send(progressQueueE, tt);
            end
        end
    else
        for tt = 1:nTasksLow
            strategy = taskListLow(tt,1);
            ss       = taskListLow(tt,2);
            s        = sList(ss);
            Delta_now = make_rescue_delta(strategy, s, Delta_damage, randomPairA, targetPair, highDelta, lowDelta);
            Cr_task_low(tt) = run_single_condition_Cr_only(Time,L,dt,RSfac_lowDFA,Noise,EnvSuccessProb,Delta_now,seed_task_low(tt));
            update_progress_bar(hProgE, nTasksLow, 'Supplement E: rescue curves');
        end
    end
    close_progress_bar(hProgE);

    Cr_random_low = nan(nS, nRepB);
    Cr_global_low = nan(nS, nRepB);
    Cr_target_low = nan(nS, nRepB);
    for tt = 1:nTasksLow
        strategy = taskListLow(tt,1);
        ss       = taskListLow(tt,2);
        rr       = taskListLow(tt,3);
        if strategy == 1
            Cr_random_low(ss,rr) = Cr_task_low(tt);
        elseif strategy == 2
            Cr_global_low(ss,rr) = Cr_task_low(tt);
        else
            Cr_target_low(ss,rr) = Cr_task_low(tt);
        end
    end

    baseLowMean = mean(Cr_base_low_rep,'omitnan');
    dmgLowMean  = mean(Cr_dmg_low_rep,'omitnan');

    figE = figure('Color','w','Units','pixels','Position',[120 120 1200 650]); hold on
    [hRandE, ~] = plot_mean_sd_band(sList, Cr_random_low, 1);
    [hGlobE, ~] = plot_mean_sd_band(sList, Cr_global_low, 2);
    [hTargE, ~] = plot_mean_sd_band(sList, Cr_target_low, 3);
    hBaseE = yline(baseLowMean, '--', sprintf('Baseline %.3f', baseLowMean), ...
        'LineWidth',1.8, 'LabelHorizontalAlignment','left');
    hDmgE  = yline(dmgLowMean,  ':',  sprintf('Damage %.3f', dmgLowMean), ...
        'LineWidth',1.8, 'LabelHorizontalAlignment','left');
    set([hBaseE hDmgE], 'FontSize',16, 'FontWeight','bold', 'FontName','Arial');
    xlabel('Rescue strength, s','FontSize',14,'FontWeight','bold');
    ylabel('Cooperation rate, C_r','FontSize',14,'FontWeight','bold');
    title(sprintf('Low-R cooperation recovery (R = %.2f, ENS = %d)', RSfac_lowDFA, ENS), ...
        'FontSize',15,'FontWeight','bold');
    hLegE = legend([hRandE hGlobE hTargE hBaseE hDmgE], ...
        {'Random/local rescue', 'Equal-budget global rescue', ...
         'CS-guided targeted rescue', 'Baseline', 'Damage'}, ...
        'Location','northwest');
    grid off; box on
    set(gca,'FontName','Arial','FontSize',13,'FontWeight','bold');
    exportgraphics(figE, pdfFile, 'ContentType','vector', 'Append', true);
end


%% ===================== SAVE RESULTS =====================
save(matFile, 'M_A','M_A_ens','Cr_A','Cr_A_sd','Cr_A_ens','Pr_A','Delta_A','condNames_A','ENS', ...
    'Cr_random','Cr_global','Cr_target','Cr_base_rep','Cr_dmg_rep','sList', ...
    'pairs','Cr_pair','Cr_pair_sd','Cr_pair_ens','csOrder','csBest','randMean','randSD', ...
    'labels','targetPair','randomPairA','highDelta','lowDelta','RSfac', ...
    'useFreshBaseSeed','fixedBaseSeed','baseSeed','seed_A','seed_B_base','seed_B_dmg','seed_task','seed_pair', ...
    'RSfac_lowDFA','makeLowRS_DFA_supplement', ...
    'MDEA_wMin','MDEA_wMax','MDEA_cbTicks','DFA_wMin','DFA_wMax','DFA_cbTicks');

fprintf('\nSaved PARFOR PDF to:\n%s\n', pdfFile);
fprintf('Saved PARFOR results MAT file to:\n%s\n', matFile);
fprintf('This run used fresh baseSeed = %d; every independent simulation used a derived distinct seed.\n', baseSeed);
toc

%% ========================================================================
%                          LOCAL FUNCTIONS
% ========================================================================

function Delta_now = make_rescue_delta(strategy, s, Delta_damage, randomPairA, targetPair, highDelta, lowDelta)
    if strategy == 1
        % Random/local rescue: spend the two-threshold local budget on the wrong pair.
        % P1/P2 remain damaged. The random pair is boosted above normal.
        Delta_now = Delta_damage;
        Delta_now(randomPairA) = highDelta + s*(highDelta - lowDelta);
    elseif strategy == 2
        % Equal-budget global rescue: same total rescue budget spread over all six channels.
        Delta_now = Delta_damage;
        totalBudget = 2*s*(highDelta - lowDelta);
        Delta_now = Delta_now + (totalBudget/6)*ones(1,6);
    else
        % CS-guided targeted rescue: same budget concentrated on P1-P2.
        Delta_now = Delta_damage;
        Delta_now(targetPair) = lowDelta + s*(highDelta - lowDelta);
    end
end




function [hMean, hBand] = plot_mean_sd_band(x, Y, styleID)
    % Y is [numel(x) x ENS]. Plot mean +/- SD and mean curve.
    % The SD band is intentionally excluded from legends; captions describe it.
    mu = mean(Y,2,'omitnan');
    sd = std(Y,0,2,'omitnan');
    x = x(:);
    mu = mu(:);
    sd = sd(:);

    switch styleID
        case 1
            marker = 'o-';
        case 2
            marker = 's-';
        otherwise
            marker = '^-';
    end

    hBand = fill([x; flipud(x)], [mu-sd; flipud(mu+sd)], [0.80 0.80 0.80], ...
        'FaceAlpha',0.33, 'EdgeColor','none', 'HandleVisibility','off');
    hMean = plot(x, mu, marker, 'LineWidth',2.2, 'MarkerSize',6);
end


function Cr_final = run_single_condition_Cr_only(Time, L, dt, RSfac, Noise, EnvSuccessProb, deltaVec, seedNow)
    % FAST simulator for Figures B and C.
    % Computes ONLY Cooperation rate Cr.
    % Does NOT store threshold time series, does NOT compute CS, MDEA, or DFA.
    % Channel order for deltaVec: [I1 T1 P1 I2 T2 P2]

    rng(seedNow, 'twister');

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

    PayS0 = zeros(2, 1);
    PayS  = zeros(2, 1);

    % Only current threshold values are needed.
    ShareInfoS1 = 1;
    TrustS1     = 1;
    SharePayS1  = 1;
    ShareInfoS2 = 1;
    TrustS2     = 1;
    SharePayS2  = 1;

    xF  = L*rand(1, 1);
    yF  = L*rand(1, 1);
    xS  = L*rand(1, 2);
    yS  = L*rand(1, 2);

    xFS = zeros(1, 3);
    yFS = zeros(1, 3);

    Cr = 0;

    for ti = 2:Time

        xFS(1) = xF(1);     yFS(1) = yF(1);
        xFS(2) = xS(1, 1);  yFS(2) = yS(1, 1);
        xFS(3) = xS(1, 2);  yFS(3) = yS(1, 2);

        thetaFS(1) = thetaF(1);
        thetaFS(2) = thetaS(1);
        thetaFS(3) = thetaS(2);

        % ---------- Prey angles ----------
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rF, L);
        list = l1FS(l2FS == 1);
        list = list(list > 1);

        if ~isempty(list)
            xSthatFpredicts = mean(xFS(list)) + velS*mean(cos(thetaFS(list)))*dt;
            ySthatFpredicts = mean(yFS(list)) + velS*mean(sin(thetaFS(list)))*dt;
            tet1 = AnglePeriodic_torus(xSthatFpredicts, ySthatFpredicts, xFS(1), yFS(1), L);
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

        % ---------- Predator angles ----------
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rS, L);

        list = l1FS(l2FS == 2);
        list = list(list <= 1);
        if ~isempty(list)
            xFthatSpredicts       = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
            yFthatSpredicts       = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
            xFthatSpredictsDECEPT = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
            yFthatSpredictsDECEPT = mean(yFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;

            if rand < 0.5
                thetaS(1) = AnglePeriodic_torus(xFthatSpredicts, yFthatSpredicts, xFS(2), yFS(2), L);
                thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(3), yFS(3), L);
            else
                thetaS(1) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(2), yFS(2), L);
                thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredicts, yFthatSpredicts, xFS(3), yFS(3), L);
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
                thetaS(2) = AnglePeriodic_torus(xFthatSpredicts, yFthatSpredicts, xFS(3), yFS(3), L);
                thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(2), yFS(2), L);
            else
                thetaS(2) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(3), yFS(3), L);
                thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredicts, yFthatSpredicts, xFS(2), yFS(2), L);
            end
        else
            thetaS(2) = thetaFS(3);
        end

        % ---------- Decisions ----------
        ShaInfo1 = rand > ShareInfoS1;
        ShaInfo2 = rand > ShareInfoS2;

        Trusted1 = 0;
        if rand > TrustS1
            Trusted1 = 1;
            thetaS(1) = thetaFoeSShared(2);
        end

        Trusted2 = 0;
        if rand > TrustS2
            Trusted2 = 1;
            thetaS(2) = thetaFoeSShared(1);
        end

        % ---------- Move ----------
        xF = mod(xF + velF * cos(thetaF) * dt, L);
        yF = mod(yF + velF * sin(thetaF) * dt, L);
        xS = mod(xS + velS * cos(thetaS) * dt, L);
        yS = mod(yS + velS * sin(thetaS) * dt, L);

        xFS = [xF(1), xS(1, 1), xS(1, 2)];
        yFS = [yF(1), yS(1, 1), yS(1, 2)];

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

        s1 = rand > SharePayS1;
        s2 = rand > SharePayS2;

        ShareUSedS1 = 0;
        ShareUSedS2 = 0;

        if Trusted1 == 1 || Trusted2 == 1
            if any_win
                ShareUSedS1 = 1;
                ShareUSedS2 = 1;

                if s1 && s2
                    PayS = [2, 2];
                    Cr = Cr + 1;
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

        % ---------- Scalar threshold updates ----------
        ShareInfoS1 = UpdateThreshold(1, ShareInfoS1, ShaInfo1, PayS(1), PayS0(1), deltaVec(1), Noise);
        TrustS1     = UpdateThreshold(1, TrustS1,     Trusted1, PayS(1), PayS0(1), deltaVec(2), Noise);
        SharePayS1  = UpdateThreshold(ShareUSedS1, SharePayS1, s1, PayS(1), PayS0(1), deltaVec(3), Noise);

        ShareInfoS2 = UpdateThreshold(1, ShareInfoS2, ShaInfo2, PayS(2), PayS0(2), deltaVec(4), Noise);
        TrustS2     = UpdateThreshold(1, TrustS2,     Trusted2, PayS(2), PayS0(2), deltaVec(5), Noise);
        SharePayS2  = UpdateThreshold(ShareUSedS2, SharePayS2, s2, PayS(2), PayS0(2), deltaVec(6), Noise);

        PayS0 = PayS;
    end

    Cr_final = Cr / Time;
end


function [signals, Ratio_Cr, Ratio_Pr] = run_single_condition_deltaVec(Time, L, dt, RSfac, Noise, EnvSuccessProb, deltaVec, seedNow, showWaitbar)

    if nargin < 9
        showWaitbar = false;
    end

    rng(seedNow, 'twister');

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

    PayS0 = zeros(2, 1);
    PayS  = zeros(2, 1);

    ShareInfoS1 = ones(Time, 1);
    ShareInfoS2 = ones(Time, 1);
    TrustS1     = ones(Time, 1);
    TrustS2     = ones(Time, 1);
    SharePayS1  = ones(Time, 1);
    SharePayS2  = ones(Time, 1);

    xF  = L*rand(1, 1);
    yF  = L*rand(1, 1);
    xS  = L*rand(1, 2);
    yS  = L*rand(1, 2);

    xFS = zeros(1, 3);
    yFS = zeros(1, 3);

    Ratio_Cr = zeros(Time, 1);
    Ratio_Pr = zeros(Time, 1);

    Cr = 0;
    TotalPay = 0;

    chunkSize = max(1, floor(Time/200));
    hWait = [];
    if showWaitbar
        try
            hWait = waitbar(0, 'Running single condition...', 'Name', 'Simulation Progress');
        catch
            hWait = [];
        end
    end

    for ti = 2:Time

        xFS(1) = xF(1);     yFS(1) = yF(1);
        xFS(2) = xS(1, 1);  yFS(2) = yS(1, 1);
        xFS(3) = xS(1, 2);  yFS(3) = yS(1, 2);

        thetaFS(1) = thetaF(1);
        thetaFS(2) = thetaS(1);
        thetaFS(3) = thetaS(2);

        % ---------- Prey angles ----------
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rF, L);
        list = l1FS(l2FS == 1);
        list = list(list > 1);

        if ~isempty(list)
            xSthatFpredicts = mean(xFS(list)) + velS*mean(cos(thetaFS(list)))*dt;
            ySthatFpredicts = mean(yFS(list)) + velS*mean(sin(thetaFS(list)))*dt;
            tet1 = AnglePeriodic_torus(xSthatFpredicts, ySthatFpredicts, xFS(1), yFS(1), L);
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

        % ---------- Predator angles ----------
        [l1FS, l2FS] = Finddistance_torus(xFS, yFS, rS, L);

        list = l1FS(l2FS == 2);
        list = list(list <= 1);
        if ~isempty(list)
            xFthatSpredicts       = mean(xFS(list)) + velF*mean(cos(thetaFoeF(list)))*dt;
            yFthatSpredicts       = mean(yFS(list)) + velF*mean(sin(thetaFoeF(list)))*dt;
            xFthatSpredictsDECEPT = mean(xFS(list)) + velF*mean(cos(thetaFoeFDECEPT(list)))*dt;
            yFthatSpredictsDECEPT = mean(yFS(list)) + velF*mean(sin(thetaFoeFDECEPT(list)))*dt;

            if rand < 0.5
                thetaS(1) = AnglePeriodic_torus(xFthatSpredicts, yFthatSpredicts, xFS(2), yFS(2), L);
                thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(3), yFS(3), L);
            else
                thetaS(1) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(2), yFS(2), L);
                thetaFoeSShared(1) = AnglePeriodic_torus(xFthatSpredicts, yFthatSpredicts, xFS(3), yFS(3), L);
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
                thetaS(2) = AnglePeriodic_torus(xFthatSpredicts, yFthatSpredicts, xFS(3), yFS(3), L);
                thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(2), yFS(2), L);
            else
                thetaS(2) = AnglePeriodic_torus(xFthatSpredictsDECEPT, yFthatSpredictsDECEPT, xFS(3), yFS(3), L);
                thetaFoeSShared(2) = AnglePeriodic_torus(xFthatSpredicts, yFthatSpredicts, xFS(2), yFS(2), L);
            end
        else
            thetaS(2) = thetaFS(3);
        end

        ShaInfo1 = 0;
        if rand > ShareInfoS1(ti), ShaInfo1 = 1; end

        ShaInfo2 = 0;
        if rand > ShareInfoS2(ti), ShaInfo2 = 1; end

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

        xF = mod(xF + velF*cos(thetaF)*dt, L);
        yF = mod(yF + velF*sin(thetaF)*dt, L);
        xS = mod(xS + velS*cos(thetaS)*dt, L);
        yS = mod(yS + velS*sin(thetaS)*dt, L);

        xFS = [xF(1), xS(1, 1), xS(1, 2)];
        yFS = [yF(1), yS(1, 1), yS(1, 2)];
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

        s1 = rand > SharePayS1(ti);
        s2 = rand > SharePayS2(ti);

        ShareUSedS1 = 0;
        ShareUSedS2 = 0;

        if Trusted1 == 1 || Trusted2 == 1
            if any_win
                ShareUSedS1 = 1;
                ShareUSedS2 = 1;

                if s1 && s2
                    PayS = [2, 2];
                    Cr = Cr + 1;
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

        % Channel order: [I1 T1 P1 I2 T2 P2]
        ShareInfoS1(ti + 1) = UpdateThreshold(1, ShareInfoS1(ti), ShaInfo1, PayS(1), PayS0(1), deltaVec(1), Noise);
        TrustS1(ti + 1)     = UpdateThreshold(1, TrustS1(ti),     Trusted1, PayS(1), PayS0(1), deltaVec(2), Noise);
        SharePayS1(ti + 1)  = UpdateThreshold(ShareUSedS1, SharePayS1(ti), s1, PayS(1), PayS0(1), deltaVec(3), Noise);

        ShareInfoS2(ti + 1) = UpdateThreshold(1, ShareInfoS2(ti), ShaInfo2, PayS(2), PayS0(2), deltaVec(4), Noise);
        TrustS2(ti + 1)     = UpdateThreshold(1, TrustS2(ti),     Trusted2, PayS(2), PayS0(2), deltaVec(5), Noise);
        SharePayS2(ti + 1)  = UpdateThreshold(ShareUSedS2, SharePayS2(ti), s2, PayS(2), PayS0(2), deltaVec(6), Noise);

        PayS0 = PayS;
        TotalPay = TotalPay + mean(PayS);
        Ratio_Pr(ti) = TotalPay / ti;
        Ratio_Cr(ti) = Cr / ti;

        if showWaitbar && (mod(ti, chunkSize) == 0 || ti == Time)
            if ~isempty(hWait) && ishandle(hWait)
                waitbar(ti/Time, hWait, sprintf('Running condition %.1f%%',100*ti/Time));
            end
            drawnow limitrate;
        end
    end

    if showWaitbar
        try
            if ~isempty(hWait) && ishandle(hWait), close(hWait); end
        catch
        end
    end

    ST0 = floor(0.25 * length(SharePayS1));
    EN0 = length(SharePayS1) - 1;

    signals = zeros(EN0 - ST0 + 1, 6);
    signals(:,1) = ShareInfoS1(ST0:EN0);
    signals(:,2) = TrustS1(ST0:EN0);
    signals(:,3) = SharePayS1(ST0:EN0);
    signals(:,4) = ShareInfoS2(ST0:EN0);
    signals(:,5) = TrustS2(ST0:EN0);
    signals(:,6) = SharePayS2(ST0:EN0);
end

function [M, P] = compute_cs_matrix_mdea(data, Slice, Newdata, str, fit_ST, fit_EN)
    nn = max(0, floor((size(data,1) - Slice) / Newdata));
    M = zeros(6,6);
    P = ones(6,6);
    Scale = zeros(max(nn,1), 6);

    if nn > 0
        for ch = 1:6
            for gg = 1:nn
                sta   = (gg - 1) * Newdata;
                DaTaa = data(1 + sta : Slice + sta, ch);
                DataX = DaTaa - min(DaTaa);
                if max(DataX) ~= 0
                    DataX = DataX ./ max(DataX);
                    Scale(gg, ch) = MDEA(DataX, str, 1, fit_ST, fit_EN, 0);
                else
                    Scale(gg, ch) = 0;
                end
            end
        end
    end

    for ii = 1:6
        for jj = 1:6
            if nn > 1
                [a, p] = corrcoef(Scale(:,ii), Scale(:,jj));
                M(ii,jj) = a(2,1);
                P(ii,jj) = p(2,1);
            else
                M(ii,jj) = 0;
                P(ii,jj) = 1;
            end
        end
    end
end


function [M, P] = compute_cs_matrix_dfa(data, Slice, Newdata, dfa_pts, dfa_order)
    nn = max(0, floor((size(data,1) - Slice) / Newdata));
    M = zeros(6,6);
    P = ones(6,6);
    Scale = zeros(max(nn,1), 6);

    if nn > 0
        for ch = 1:6
            for gg = 1:nn
                sta   = (gg - 1) * Newdata;
                DaTaa = data(1 + sta : Slice + sta, ch);
                if max(DaTaa) ~= 0
                    aa = DFA_func(DaTaa, dfa_pts, dfa_order, 0);
                    Scale(gg, ch) = aa(1);
                else
                    Scale(gg, ch) = 0;
                end
            end
        end
    end

    for ii = 1:6
        for jj = 1:6
            if nn > 1
                [a, p] = corrcoef(Scale(:,ii), Scale(:,jj));
                M(ii,jj) = a(2,1);
                P(ii,jj) = p(2,1);
            else
                M(ii,jj) = 0;
                P(ii,jj) = 1;
            end
        end
    end
end

function [A,F] = DFA_func(data, pts, order, PLOT)
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
        scatter(log10(pts), log10(F)); hold on
        x = pts;
        plot(log10(x), polyval(A, log10(x)), '--')
        xlabel('log_{10} W'); ylabel('log_{10} F(W)');
        legend(['\alpha = ' num2str(sprintf('%.3f', A(1)))], 'Location', 'northwest');
        hold off
    end
end

function plot_cs_network(M, labels, panelTitle, wMin, wMax, cbTicks)
    lineWidth  = 4.0;
    nodeSize   = 220;
    nodeFS     = 12;
    edgeFS     = 10;
    nCurvePts  = 180;

    tLabelsCross = [0.40 0.55 0.70; 0.38 0.58 0.72; 0.42 0.52 0.68];
    labelPushCross = [0.02 0.05 0.08; 0.03 0.06 0.09; 0.02 0.05 0.08];
    % Internal same-agent labels.
    % Use mirrored label placement so every left-side internal edge has a
    % geometrically symmetric right-side counterpart.  This keeps the CS
    % numbers at the same relative distance from their corresponding curves:
    %   I1-T1 mirrors I2-T2
    %   T1-P1 mirrors T2-P2
    %   I1-P1 mirrors I2-P2
    % The long outer arcs use the outside of the curve to avoid overlap with
    % the middle node labels, but the distance from the line is identical.
    tInternal = 0.50;
    labelGapInternal = 0.055;
    pushIntLeft  = [ labelGapInternal;  labelGapInternal; -labelGapInternal];   % [I1-T1; T1-P1; I1-P1]
    pushIntRight = [-labelGapInternal; -labelGapInternal;  labelGapInternal];   % [I2-T2; T2-P2; I2-P2]

    xSep = 2.0;
    offScale = xSep;
    x = [0 0 0  xSep xSep xSep];
    y = [3 2 1  3    2    1];

    tCrossBase = [0.12 0.22 0.32; 0.16 0.26 0.36; 0.10 0.20 0.30];
    crossOffsets = offScale * tCrossBase;
    leftPairs = [1 2; 2 3; 1 3];
    leftOffs  = offScale * [0.12; 0.12; -0.34];
    rightPairs = [4 5; 5 6; 4 6];
    rightOffs  = offScale * [-0.12; -0.12; 0.34];

    hold on
    cmap = flipud(gray(256));
    colormap(cmap);
    caxis([wMin wMax]);
    imagesc([wMin wMax], [0 0], [wMin wMax]);
    set(gca,'YDir','normal');

    scatter(x(1:3), y(1:3), nodeSize, 'k', 'filled');
    scatter(x(4:6), y(4:6), nodeSize, 'k', 'filled');

    for i = 1:3
        text(x(i)-0.18, y(i), labels{i}, 'HorizontalAlignment','right', 'VerticalAlignment','middle', ...
            'FontSize',nodeFS, 'FontWeight','bold');
    end
    for i = 4:6
        text(x(i)+0.18, y(i), labels{i}, 'HorizontalAlignment','left', 'VerticalAlignment','middle', ...
            'FontSize',nodeFS, 'FontWeight','bold');
    end

    for iL = 1:3
        for jR = 4:6
            drawEdgeLabelOnCurve(gca, M, x, y, iL, jR, crossOffsets(iL,jR-3), nCurvePts, ...
                wMin, wMax, lineWidth, edgeFS, tLabelsCross(iL,jR-3), labelPushCross(iL,jR-3));
        end
    end
    for k = 1:3
        drawEdgeLabelOnCurve(gca, M, x, y, leftPairs(k,1), leftPairs(k,2), leftOffs(k), ...
            nCurvePts, wMin, wMax, lineWidth, edgeFS, tInternal, pushIntLeft(k));
    end
    for k = 1:3
        drawEdgeLabelOnCurve(gca, M, x, y, rightPairs(k,1), rightPairs(k,2), rightOffs(k), ...
            nCurvePts, wMin, wMax, lineWidth, edgeFS, tInternal, pushIntRight(k));
    end

    axis off
    xlim([-0.5, xSep + 0.5]);
    ylim([0.5 3.5]);
    title(panelTitle, 'FontSize',14, 'FontWeight','bold');
    cb = colorbar;
    cb.Label.String   = 'CS';
    cb.Label.FontSize = 12;
    cb.Ticks          = cbTicks;
    hold off
end

function drawEdgeLabelOnCurve(ax, M, x, y, i, j, off, nPts, wMin, wMax, lw, fs, tLab, push)
    w = M(i,j);
    if ~isfinite(w), return; end
    s = (min(max(w, wMin), wMax) - wMin) / (wMax - wMin);
    col = (1 - s)*[1 1 1] + s*[0 0 0];
    [xc, yc, xl, yl, ang, nx, ny] = quadCurveLabelAngle(x(i), y(i), x(j), y(j), off, nPts, tLab);
    plot(ax, xc, yc, 'LineWidth', lw, 'Color', col)
    xl = xl + push*nx;
    yl = yl + push*ny;
    if ang > 90, ang = ang - 180; end
    if ang < -90, ang = ang + 180; end
    text(ax, xl, yl, sprintf('%.2f', w), 'FontSize', fs, 'Rotation', ang, ...
        'HorizontalAlignment','center', 'VerticalAlignment','middle', 'BackgroundColor','w', 'Margin',1);
end

function [xc, yc, xl, yl, ang, nx, ny] = quadCurveLabelAngle(x1, y1, x2, y2, off, nPts, t)
    dx = x2 - x1;
    dy = y2 - y1;
    Ld = hypot(dx, dy);
    nx0 = -dy / Ld;
    ny0 =  dx / Ld;
    cx = (x1 + x2)/2 + off*nx0;
    cy = (y1 + y2)/2 + off*ny0;
    tt = linspace(0, 1, nPts);
    xc = (1 - tt).^2*x1 + 2*(1 - tt).*tt*cx + tt.^2*x2;
    yc = (1 - tt).^2*y1 + 2*(1 - tt).*tt*cy + tt.^2*y2;
    xl = (1 - t)^2*x1 + 2*(1 - t)*t*cx + t^2*x2;
    yl = (1 - t)^2*y1 + 2*(1 - t)*t*cy + t^2*y2;
    dBx = 2*(1 - t)*(cx - x1) + 2*t*(x2 - cx);
    dBy = 2*(1 - t)*(cy - y1) + 2*t*(y2 - cy);
    ang = atan2d(dBy, dBx);
    Lder = hypot(dBx, dBy);
    nx = -dBy / Lder;
    ny =  dBx / Lder;
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
            dp = 1 * ChangeThreshold;
        end
        if Pay ~= 0 && Paybefore ~= 0
            DeltaPay = Pay - Paybefore;
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
    if max(Data) > 0
        Data = Data ./ max(Data);
    end

    Lengthdata = length(Data);
    Ddata      = Data ./ Stripesize;
    Event      = zeros(Lengthdata, 1);

    k = 1;
    Event(1) = 1;
    StartEvent = zeros(Lengthdata,1);

    for i = 2:Lengthdata
        if Ddata(i) < floor(Ddata(i-1)) || Ddata(i) > ceil(Ddata(i-1))
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
                if rand < 0.5, State0(yy) = 1; else, State0(yy) = -1; end
            end
        end
        Diff = cumsum(State0);
    else
        State0 = zeros(Lengthdata, 1);
        for yy = 1:Lengthdata
            if Event(yy) == 1
                if rand < 0.5, State0(yy) = 1; else, State0(yy) = -1; end
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

    ll = floor(log(length(Diff))/log(1.2)) - 5;
    Delh = zeros(1, ll);
    de   = zeros(1, ll);
    DE   = zeros(1, ll);

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
        while i <= enn2 && StartEvent(i) + del < enn
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
        DE(:, q) = -sum(Pc .* log(Pc) / log(10));
    end

    for t = 1:length(Delh)
        de(t) = log(Delh(t)) / log(10);
    end

    Starr = max(1, round(ST * length(de)));
    endd  = max(Starr, round(EN * length(de)));
    DE0 = DE(Starr:endd);
    de0 = de(Starr:endd);
    FitLine = polyfit(de0, DE0, 1);
    delta = FitLine(1);

    if PLOT == 1
        figure
        subplot(1, 2, 1)
        plot(Ddata)
        xlabel('t')
        ylabel('X(t)')
        title('Signal')
        subplot(1, 2, 2)
        plot(de(3:max(3,length(de)-3)), DE(3:max(3,length(DE)-3)), '+')
        hold on
        plot(de0, FitLine(1)*de0 + FitLine(2), 'r--', 'LineWidth', 1.5)
        xlabel('log(l)')
        ylabel('S(l)')
        legend(['\delta = ' num2str(sprintf('%.3f', delta))], 'Location', 'northwest')
    end
end



function h = init_progress_bar(enabled, msg, total)
    % Robust progress monitor for both FOR and PARFOR.
    % It always prints progress to the Command Window. If MATLAB allows GUI
    % figures, it also opens a waitbar. The waitbar can now be closed safely
    % by the user, and stale waitbar windows are deleted instead of getting
    % stuck after PARFOR/DataQueue callbacks finish.
    h = struct('enabled',enabled,'msg',msg,'total',total,'key','','waitHandle',[]);
    if ~enabled
        return
    end

    h.key = matlab.lang.makeValidName(sprintf('%s_%0.0f', msg, 1e9*rand));
    set_progress_counter(h.key, 0);

    fprintf('\n%s: 0/%d (0.0%%)\n', msg, total);

    try
        % Delete any previous waitbar with the same title before opening a new one.
        oldBars = findall(0, 'Type', 'figure', 'Tag', 'TMWWaitbar', 'Name', msg);
        if ~isempty(oldBars)
            delete(oldBars);
        end

        h.waitHandle = waitbar(0, sprintf('%s: 0/%d (0.0%%)', msg, total), ...
            'Name', msg, ...
            'CreateCancelBtn', 'setappdata(gcbf,''canceling'',1); delete(gcbf);', ...
            'CloseRequestFcn', 'setappdata(gcbf,''canceling'',1); delete(gcbf);');
        setappdata(h.waitHandle, 'canceling', 0);
        drawnow limitrate;
    catch
        h.waitHandle = [];
        fprintf('%s: GUI waitbar unavailable; using Command Window progress only.\n', msg);
    end
end

function update_progress_bar(h, total, msg)
    if isempty(h) || ~isstruct(h) || ~isfield(h,'enabled') || ~h.enabled
        return
    end

    c = increment_progress_counter(h.key);
    frac = min(c/total, 1);

    % Print/update at every completed task for small task counts, otherwise
    % roughly 5%% increments. This makes progress visible even in PARFOR.
    step = max(1, floor(total/20));
    doUpdate = (c == 1) || (mod(c, step) == 0) || (c >= total);

    if doUpdate
        fprintf('%s: %d/%d (%.1f%%)\n', msg, c, total, 100*frac);

        try
            if isfield(h,'waitHandle') && ~isempty(h.waitHandle) && isgraphics(h.waitHandle)
                waitbar(frac, h.waitHandle, sprintf('%s: %d/%d (%.1f%%)', ...
                    msg, c, total, 100*frac));
                drawnow limitrate;
            end
        catch
            % Command Window progress already printed; no need to stop run.
        end
    end
end

function close_progress_bar(h)
    if isempty(h) || ~isstruct(h)
        return
    end

    try
        if isfield(h,'waitHandle') && ~isempty(h.waitHandle) && isgraphics(h.waitHandle)
            waitbar(1, h.waitHandle, sprintf('%s: complete', h.msg));
            drawnow limitrate;
            delete(h.waitHandle);
        end
    catch
        % If the user already closed the waitbar, ignore it.
    end

    fprintf('%s: complete.\n', h.msg);
end

function cleanup_all_progress_bars()
    % Emergency cleanup for stale MATLAB waitbar windows. This is called at
    % the start of the script and automatically again when the script exits.
    try
        oldBars = findall(0, 'Type', 'figure', 'Tag', 'TMWWaitbar');
        if ~isempty(oldBars)
            delete(oldBars);
        end
    catch
    end
end

function set_progress_counter(key, value)
    persistent counts
    if isempty(counts)
        counts = containers.Map('KeyType','char','ValueType','double');
    end
    counts(key) = value;
end

function c = increment_progress_counter(key)
    persistent counts
    if isempty(counts)
        counts = containers.Map('KeyType','char','ValueType','double');
    end
    if ~isKey(counts,key)
        counts(key) = 0;
    end
    counts(key) = counts(key) + 1;
    c = counts(key);
end
