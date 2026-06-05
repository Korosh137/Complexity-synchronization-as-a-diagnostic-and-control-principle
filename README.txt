README

Complexity Synchronization as a Diagnostic and Control Principle for Adaptive Systems

Author: Korosh Mahmoodi

This repository contains the MATLAB codes used to generate all main and
supplementary figures reported in the paper:

“Complexity Synchronization as a Diagnostic and Control Principle for
Adaptive Systems”

OVERVIEW

The codes implement simulations of a reduced Predator–Prey model
composed of adaptive Selfish Algorithm agents. The repository includes
routines for:

-   Adaptive multi-agent simulations
-   Complexity Synchronization (CS) analysis
-   Modified Diffusion Entropy Analysis (MDEA)
-   Detrended Fluctuation Analysis (DFA)
-   Renewal and autocorrelation analyses
-   Perturbation and rescue experiments
-   Ordinary correlation control analyses
-   Figure generation for all manuscript figures

REPOSITORY STRUCTURE

Main manuscript figures: Figures 3–17

Supplementary analyses: Figure 18 and supporting control analyses

Core methods: - Selfish Algorithm (SA) - Complexity Synchronization
(CS) - Modified Diffusion Entropy Analysis (MDEA) - Detrended
Fluctuation Analysis (DFA) - Renewal analysis - Perturbation and rescue
experiments

CORE FUNCTIONS

The figure scripts rely on several shared analysis and utility
functions. These files should also be included in the repository.

Required analysis functions: MDEA.m Modified Diffusion Entropy Analysis
used throughout the manuscript.

    DFA_func.m
        Detrended Fluctuation Analysis used throughout the manuscript.

Supporting utility functions: Include any custom helper functions
required by the figure scripts, such as geometric, learning, renewal,
synchronization, or plotting routines (e.g., UpdateThreshold.m,
Finddistance_torus.m, AnglePeriodic_torus.m, and related dependencies).

FILES

Fig_3_Animation.m Animation of the Predator–Prey model dynamics.

Fig_4_trajectories.m Example trajectories of adaptive threshold
variables.

Fig_5a_CCPayvsRS.m Cooperation and payoff versus sensing radius.

Fig_5b_CCPayTime.m Time evolution of cooperation and payoff.

Fig_6a_CStimeMDEAgraph.m Example MDEA scaling time series and scaling
plots.

Fig_6b_CStimeDFAgraph.m Example DFA scaling time series and scaling
plots.

Fig_7_CvsCS_MDEA.m Cooperation versus MDEA-based complexity
synchronization.

Fig_8_CvsCS_DFA.m Cooperation versus DFA-based complexity
synchronization.

Fig_9_RenewalTest.m Renewal aging analysis.

Fig_10_AutocorrelationTaus.m Inter-event interval autocorrelation
analysis.

Fig_11_CSperturbation_MDEA.m Perturbation analysis using MDEA-based CS.

Fig_12_CSperturbation_DFA.m Perturbation analysis using DFA-based CS.

Fig_13_14_15_16_17.m CS-guided rescue, intervention, and network
analyses.

Fig_18_CrossCorrelationRaw.m Raw Pearson-correlation control analysis.

HOW TO REPRODUCE THE FIGURES

Run the MATLAB scripts corresponding to the figure numbers in the
manuscript.

Main figures: Fig_3_Animation.m -> Figure 3 Fig_4_trajectories.m ->
Figure 4 Fig_5a_CCPayvsRS.m -> Figure 5 (left panel) Fig_5b_CCPayTime.m
-> Figure 5 (right panel) Fig_6a_CStimeMDEAgraph.m -> Figure 6 (MDEA)
Fig_6b_CStimeDFAgraph.m -> Figure 6 (DFA) Fig_7_CvsCS_MDEA.m -> Figure 7
Fig_8_CvsCS_DFA.m -> Figure 8 Fig_9_RenewalTest.m -> Figure 9
Fig_10_AutocorrelationTaus.m -> Figure 10 Fig_11_CSperturbation_MDEA.m
-> Figure 11 Fig_12_CSperturbation_DFA.m -> Figure 12
Fig_13_14_15_16_17.m -> Figures 13–17 Fig_18_CrossCorrelationRaw.m ->
Supplementary control analysis

Expected runtime: Short figures (minutes): Figures 3, 4, 6, 9, and 10.

    Moderate figures (tens of minutes):
        Figures 5, 7, 8, 11, and 12.

    Long publication-scale simulations (hours depending on hardware):
        Figures 13–17 and large ensemble sweeps.

REQUIREMENTS

-   MATLAB
-   Parallel Computing Toolbox (optional for faster execution)
-   Sufficient memory for long simulations (up to 10^6 trials)

NOTES

-   Several scripts require long runtimes because they reproduce
    publication-scale simulations.
-   Random-number generators are initialized to produce independent
    realizations.
-   Figure styles were standardized to match the published manuscript.
-   For exact reproducibility, users may wish to fix random seeds.
-   Simulation parameters are documented within each script.
-   Before publishing or reproducing results, verify that all required
    helper functions are present in the repository and accessible from
    the MATLAB path.

CITATION

If you use these codes, please cite the associated publication:

Korosh Mahmoodi et al. “Complexity Synchronization as a Diagnostic and
Control Principle for Adaptive Systems”

Copyright (c) Korosh Mahmoodi.
