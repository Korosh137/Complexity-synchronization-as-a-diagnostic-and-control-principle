function delta = MDEA(Data, Stripesize, Rule, ST, EN, PLOT)

% This function measures the scaling (delta) of the time series (Data)
% using Modified Diffusion Entropy Analysis (MDEA).
%
% Data: time series of interest. Minimum length is about 3e4 data points.
% Stripesize: size of the stripes for Data normalized to [0, 1].
% Suggested Stripesize is between 0.001 and 0.01.
%
% Rule:
%   Rule = 1  : assigns +1 to events and accumulates them.
%   Rule = -1 : assigns randomly +1 or -1 to events and accumulates them.
%   Rule = 0  : fills regions between events by +1 or -1 randomly and accumulates them.
%
% ST: beginning of linear estimate, suggested ST = 0.4.
% EN: ending of linear estimate, suggested EN = 0.8.
% PLOT: if 1, plots MDEA graph.
%
% Cite:
% Mahmoodi, Korosh, et al. "Complexity synchronization: a measure of
% interaction between the brain, heart and lungs." Scientific Reports 13.1
% (2023): 11433.

%%% Normalizing the Data to [0 1]
Data = Data - min(Data);
Data = Data ./ max(Data);

LengthData = length(Data);

%%% Extracting events using stripes
%%% Each time Data passes from one stripe to another, it is recorded as an event.

Event = zeros(LengthData, 1);
Event(1) = 1;

StartEvent = [];
k = 1;

stripe_prev = floor(Data(1) / Stripesize);

for i = 2:LengthData

    stripe_now = floor(Data(i) / Stripesize);

    if stripe_now ~= stripe_prev
        Event(i) = 1;
        StartEvent(k) = i;
        k = k + 1;
    end

    stripe_prev = stripe_now;

end

%%% Creating diffusion trajectory Diff from extracted events

StartEvent = StartEvent(StartEvent ~= 0);

if Rule == 1
    Diff = cumsum(Event);
end

if Rule == -1

    State0 = zeros(LengthData, 1);

    for yy = 1:LengthData

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

    State0 = zeros(LengthData, 1);

    for yy = 1:LengthData

        if Event(yy) == 1
            r = rand;

            if r < 0.5
                State0(yy) = 1;
            else
                State0(yy) = -1;
            end
        end

    end

    State00 = zeros(LengthData, 1);
    State0(1) = 1;

    for ee = 2:LengthData

        if State0(ee) == 0
            State00(ee) = State00(ee - 1);
        else
            State00(ee) = State0(ee);
        end

    end

    Diff = cumsum(State00);

end

%%% Evaluating Shannon entropy of the diffusion trajectory Diff

ll = floor(log(length(Diff)) / log(1.2)) - 5;

Delh = zeros(1, ll);
de = zeros(1, ll);
DE = zeros(1, ll);

for i = 1:ll
    Delh(i) = floor(1.2^i);
end

for q = 1:length(Delh)

    SliceNum = length(StartEvent);

    del = Delh(q);

    HH = zeros(SliceNum, 1);

    enn = length(Diff);
    enn2 = length(StartEvent);

    i = 1;

    while StartEvent(i) + del < enn

        idx0 = StartEvent(i);
        HH(i) = Diff(idx0 + del) - Diff(idx0);

        i = i + 1;

        if i == enn2
            break;
        end

    end

    XF = HH(HH ~= 0);

    %%% Cleaner entropy binning using integer-valued diffusion displacement
    XF_round = round(XF);
    binEdges = (min(XF_round) - 0.5):(max(XF_round) + 0.5);

    counts = histcounts(XF_round, binEdges);

    counts = counts(counts ~= 0);
    Pc = counts ./ sum(counts);

    DE0 = -sum(Pc .* log10(Pc));

    DE(:, q) = DE0;

end

for t = 1:length(Delh)
    de(t) = log10(Delh(t));
end

Starr = round(ST * length(de));
endd = round(EN * length(de));

DE0 = DE(Starr:endd);
de0 = de(Starr:endd);

%%% Linear fit to estimate delta
FitLine = polyfit(de0, DE0, 1);
delta = FitLine(1);

if PLOT == 1

    figure

    subplot(1, 2, 1)
    plot(Data ./ Stripesize)
    xlabel('t')
    ylabel('X(t)')
    legend('X(t)', 'Location', 'northwest')
    title('Signal')

    subplot(1, 2, 2)
    plot(de(3:length(de)-3), DE(3:length(DE)-3), '+', ...
         de0, FitLine(1) * de0 + FitLine(2), 'r--', ...
         'LineWidth', 1.5)

    xlabel('log(l)')
    ylabel('S(l)')

    legend(['\delta = ' num2str(sprintf('%.3f', delta))], ...
           'Location', 'northwest')

end

end