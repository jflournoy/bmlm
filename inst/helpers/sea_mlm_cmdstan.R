library(data.table)
library(rstan)
#setwd('~/otherhome/code/bmlm/')
setwd('~/code_new/bmlm/')

##!!!!
# Need to ensure data is ordered by ROI for map_rect
###

load('data/sea_fearGTcalm_stress_psych.rda')
sea_fearGTcalm_stress_psych[, id_idx := as.numeric(as.factor(sea_fearGTcalm_stress_psych$idnum))]
sea_fearGTcalm_stress_psych[, roi_idx := as.numeric(as.factor(sea_fearGTcalm_stress_psych$roi))]
names(sea_fearGTcalm_stress_psych)

setorder(sea_fearGTcalm_stress_psych, roi_idx, id_idx)
sea_fearGTcalm_stress_psych[1:50, .(roi_idx, id_idx, time)]
#for testing, select random set of ROIs
set.seed(2242211)
test_rois <- sample(sea_fearGTcalm_stress_psych[, unique(roi_idx)], 10)

###

# x -> m
# fearGTcalm_WCEN ~ 1 + TIMECENTER + WCEN_EPISODICTOT + GCEN_EPISODICTOT
#
# x, m -> y
# GAD7_TOT ~ 1 + TIMECENTER + WCEN_EPISODICTOT + GCEN_EPISODICTOT + fearGTcalm_WCEN + fearGTcalm_GCEN

d = copy(sea_fearGTcalm_stress_psych[roi_idx %in% test_rois])
#d = copy(sea_fearGTcalm_stress_psych)
id = "id_idx"
roi = "roi_idx"
x = "WCEN_EPISODICTOT"
m = "fearGTcalm_WCEN"
y = "GAD7_TOT_lead"
time = 'TIMECENTER'
covars_m <- c('GCEN_EPISODICTOT')
covars_y <- c('GCEN_EPISODICTOT', 'fearGTcalm_GCEN')
d_cols <- unique(c(id, roi, x, m, y, time, covars_m, covars_y))

priors = NULL
binary_y = FALSE

#cleanup d
if(class(d)[1] == "data.table"){
    d <- na.omit(d[, d_cols, with = FALSE])
    d <- as.data.frame(d)
} else {
    if (class(d)[1] == "tbl_df") d <- as.data.frame(d)  # Allow tibbles
    d <- na.omit(d[, dcols])
}

# Check priors
default_priors <- list(
    bs = 1000, id_taus = 50, roi_taus = 50,
    id_lkj_shape = 1, roi_lkj_shape = 1,
    ybeta = 1000,
    mbeta = 1000,
    sigmas = 1
)
if (is.null(priors$bs)) priors$bs <- default_priors$bs
if (is.null(priors$id_taus)) priors$id_taus <- default_priors$id_taus
if (is.null(priors$id_lkj_shape)) priors$id_lkj_shape <- default_priors$id_lkj_shape
if (is.null(priors$roi_taus)) priors$roi_taus <- default_priors$roi_taus
if (is.null(priors$roi_lkj_shape)) priors$roi_lkj_shape <- default_priors$roi_lkj_shape
if (is.null(priors$mbeta)) priors$mbeta <- default_priors$mbeta
if (is.null(priors$ybeta)) priors$ybeta <- default_priors$ybeta
if (is.null(priors$sigmas)) priors$sigmas <- default_priors$sigmas
names(priors) <- lapply(names(priors), function(x) paste0("prior_", x))

# Create a data list for Stan
ld <- list()
ld$id = as.integer(as.factor(d[,id]))  # Sequential IDs
ld$roi = as.integer(as.factor(d[,roi]))  # Sequential IDs
ld$X = d[,x]
ld$M = d[,m]
ld$Y = d[,y]
ld$Time = d[,time]
ld$Cy = as.matrix(d[, covars_y])
ld$Cm = as.matrix(d[, covars_m])
ld$N <- nrow(d)
ld$J <- length(unique(ld$id))
ld$K <- length(unique(ld$roi))
ld$Ly <- ncol(ld$Cy)
ld$Lm <- ncol(ld$Cm)
ld <- append(ld, priors)
ld$SIMULATE <- 0

N_per_roi <- unlist(lapply(split(ld$X, ld$roi), function(x) sum(!is.na(x))))
if(all(max(N_per_roi) - min(N_per_roi) == 0)){
    message(N_per_roi[[1]], ' data points for each ROI. Continuing...')
} else if(MAPRECT){
    stop('Cannot continue, unequal data size for some ROIs!')
}

# Write data

stan_rdump(ls(ld), file.path('~/code_new/cmdstan/roi_hlm', "roi_hlm_input.R"), envir = list2env(ld))

# Simulate
# default_priors <- list(
#     bs = 1000, id_taus = 50, roi_taus = 50,
#     id_lkj_shape = 1, roi_lkj_shape = 1,
#     ybeta = 1000,
#     mbeta = 1000
# )
ld_sim <- ld
ld_sim$SIMULATE <- 1
ld_sim$prior_bs <- ld_sim$prior_mbeta <- ld_sim$prior_ybeta <-  1
ld_sim$prior_id_taus <- ld_sim$prior_roi_taus <- 1

stan_rdump(ls(ld_sim), file.path('~/code_new/cmdstan/roi_hlm', "roi_hlm_input_for_sim.R"), envir = list2env(ld_sim))

#./bmlm_map_rect sample num_samples=1 num_warmup=500 output file=sim.csv data file=roi_hlm_input_for_sim.R
library(rstan)
library(tidybayes)
library(tidyr)
sim_data <- rstan::read_stan_csv(file.path('~/code_new/cmdstan/roi_hlm', 'sim.csv'))
y_sim <- tidybayes::gather_draws(sim_data, `Y_sim.*`, regex = TRUE) %>%
    tidyr::extract(.variable, into = c('variable', 'id'), regex = '(Y_sim)\\.(.*)')
m_sim <- tidybayes::gather_draws(sim_data, `M_sim.*`, regex = TRUE) %>%
    tidyr::extract(.variable, into = c('variable', 'id'), regex = '(M_sim)\\.(.*)')
params <- tidybayes::gather_draws(sim_data, `(.*_(a|b|cp).*|(a|b|cp))`, regex = TRUE)

ld_from_sim <- ld
ld_from_sim$Y <- y_sim$.value
ld_from_sim$M <- m_sim$.value

stan_rdump(ls(ld_from_sim), file.path('~/code_new/cmdstan/roi_hlm', "roi_hlm_input_from_sim.R"), envir = list2env(ld_from_sim))

#export STAN_NUM_THREADS=7
#./bmlm_map_rect sample num_samples=1000 num_warmup=1000 output file=fit_from_sim.csv data file=roi_hlm_input_from_sim.R

#Transform data

attach(ld)

P = 7;                      # Number of person & ROI-varying variables: dm, dy, a, b, cp, ty, tm
#   That is, intercept for m and y equations, a path, b path, c prime path,
#   and 2 time effects.
n_int = 7;                   # Extra ints we need for the function
Nr = 4 + Ly + Lm;           # Number of real-valued variables: Y, X, M, Time and the covariates
Ni = 1;                     # Number of int-valued variables
slen = N / K;    # Number of observations per variable per shard, assumes equal number of obs per K
x_r = array(dim = c(K, slen * Nr));         # Array with a row per shard, and enough columns to hold all the real data
x_i = array(dim = c(K, slen * Ni + n_int)); # Ditto, interger data.

# Make Shards, one per K rois

x_i[,1] = rep(slen, K);
x_i[,2] = rep(P, K);
x_i[,3] = rep(Nr, K);
x_i[,4] = rep(Ni, K);
x_i[,5] = rep(Ly, K);
x_i[,6] = rep(Lm, K);
x_i[,7] = rep(J, K);
for (k in 1:K){
    beg = 1 + (k-1)*slen; #k = 1, beg = 1
    end = k*slen;         #k = 1, end = slen = N/K
    x_r[k, (0*slen+1):(1*slen)] = (Y[ beg:end ]);
    x_r[k, (1*slen+1):(2*slen)] = (X[ beg:end ]);
    x_r[k, (2*slen+1):(3*slen)] = (M[ beg:end ]);
    x_r[k, (3*slen+1):(4*slen)] = (Time[ beg:end ]);

    #matrix[N, Ly] Cy;
    for (lyi in 1:Ly){
        x_r[k, ((4+lyi-1)*slen+1):((4+lyi)*slen)] = (Cy[ beg:end, lyi ]);
    }
    #matrix[N, Lm] Cm;
    for (lmi in 1:Lm){
        x_r[k, ((4+Ly+lmi-1)*slen+1):((4+Ly+lmi)*slen)] = (Cm[ beg:end, lmi ]);
    }
    #int<lower=1,upper=J> id[N];
    x_i[k, (n_int + 1):(slen * Ni + n_int)] = ld$id[ beg:end ];
}

funcs <- rstan::expose_stan_functions('exec/bmlm_map_rect.stan')

sea_fearGTcalm_stress_psych[roi == sort(test_rois)[1],get(y)]
round(na.omit(sea_fearGTcalm_stress_psych)[roi == sort(test_rois)[1],get(x)],6)
round(na.omit(sea_fearGTcalm_stress_psych)[roi == sort(test_rois)[10],get(covars_y[2])],6)

hlm_med(global = 1:(P+Lm+Ly+J*P+2), Vs = 1:P, xr = x_r[10,], xi = x_i[10,])

mm_ <- c(-1.26356,1.23342,-0.144721,-3.24918,1.89895,1.58645,0.547887,2.2022,-2.37525,-0.436182,0.917747,-2.26589,0.955775,-0.0201654,0.685714,1.08192,-0.955309,0.603873,-0.953799,-0.0498699,0.127372,0.657838,0.404036,0.649823,0.168793,-0.108841,-0.718626,0.441293,-1.68149,0.0598032,0.163299,1.2325,-0.0897857,-0.0247131,-0.673713,-0.683736,-0.0813618,0.59111,0.891074,-1.32467,-0.00787478,0.17074,0.397163,-0.284859,-1.08141,0.730133,1.52862,-1.95591,0.387321,-1.98652,2.16211,0.280089,0.249845,-1.80592,-1.1658,1.45538,-0.722127,1.14563,-0.762923,-0.218925,0.198254,-0.111273,0.334834,0.605079,0.447396,0.228575,0.202939,-0.923954,-0.724618,0.824736,-1.23279,0.675195,-0.251966,0.160057,-0.788524,1.12847,0.817076,-0.607634,-0.124939,1.45222,-0.534623,0.221807,-1.0509,0.498458,0.533918,-1.02919,1.55936,-1.52611,-0.363303,1.02974,-0.588498,1.62495,-2.28832,-0.817581,0.149758,-1.52601,1.92325,0.856019,-0.224921,-0.743578,-0.229986,0.81491,1.20188,-1.25853,1.06438,-0.457223,-0.881049,1.87643,0.843997,0.861245,1.01883,-0.721013,-1.54298,-1.85963,0.303024,-2.35057,-0.73241,-0.37188,1.55441,2.09139,-0.578028,0.134621,0.455618,-0.506171,-0.801205,-0.806576,1.20201,0.367522,0.44985,1.60093,-0.837106,-0.971937,-0.0854694,0.654842,0.92823,-0.213444,0.032076,-0.425989,-0.436053,-0.15961,-0.411699,0.117115,0.8908,0.673153,-1.78078,1.08692,-1.05192,1.12187,0.212647,-1.44851,-0.212122,0.507946,-2.19453,1.28223,-0.213183,1.1221,-0.981816,1.751,-0.702518,0.746677,-0.463811,-0.346146,1.79932,-0.502193,-1.25407,-0.211787,-1.05488,0.888342,0.705226,-0.547479,0.177533,0.838197,-1.04712,-0.625831,0.287199,0.301969,-1.25081,-0.504956,0.526508,0.178252,1.29658,-0.104048,-1.12012,1.09441,0.196437,1.00781,-0.289163,-1.08756,0.0395579,-0.285299,0.547967,0.629743,1.38796,-0.00409261,-1.47547,0.684166,-1.51756,-0.00109926,0.620766,-1.18799,0.863582,0.812959,0.892391,-2.13997,-0.0141519,1.06346,-0.614681,-0.365904,-0.323309,1.52441,-1.37864,-0.337022,1.24591,1.73699,-1.94944,-0.0328412,-0.120137,-0.098109,0.326983,-0.473265,0.449452,-1.06229,0.847649,0.453267,-0.441872,1.70073,-1.70255,-2.06866,-0.273286,0.0819429,1.02377,1.1173,0.353544,1.30872,0.387813,-0.641596,-1.28955,0.905181,0.0612104,-1.16333,-0.926666,0.34303,-0.870844,-0.42873,1.2102,0.390052,0.479899,-0.961379,-0.59894,1.82359,-0.001127,-0.464674,0.70854,-0.271104,-0.849896,0.170031,0.444957,-0.487457,-0.492556,0.457669,-1.63625,-0.97083,-0.200839,1.31036,1.61096,0.112477,0.296465,0.625292,-0.785632,-0.172631,-0.78645,1.98905,2.24362,-0.110315,-1.6744,0.540989,-1.86952)


xx_ <- c(0.138889,-0.361111,-0.361111,2.63889,3.63889,4.13889,-2.36111,-0.861111,-2.36111,-0.361111,-1.58333,-0.0833333,0.416667,2.41667,1.41667,-1.58333,3.41667,-1.58333,0.416667,-1.58333,1.79167,3.29167,-1.70833,-1.70833,4.79167,-1.70833,0.291667,0.291667,-0.208333,-1.70833,2.54167,1.54167,-1.45833,1.54167,1.54167,0.541667,-1.45833,1.04167,-1.45833,-1.45833,-2.90909,1.09091,1.59091,-0.909091,-2.90909,1.09091,-0.409091,5.09091,1.10417,1.10417,1.10417,-0.895833,-0.895833,-0.895833,1.10417,1.10417,-0.895833,-0.895833,-1.33333,-1.33333,5.66667,-0.833333,5.16667,1.16667,-0.833333,-3.33333,-3.33333,-0.833333,-1.95833,2.04167,-1.95833,-1.95833,2.04167,-1.95833,2.54167,3.04167,0.0416667,0.0416667,0.0416667,-1.95833,3.04167,-1.95833,1.04167,0.0416667,0.0416667,-0.458333,-1.95833,4.04167,3.58333,1.08333,-0.416667,-1.91667,0.0833333,-1.91667,7.08333,-1.91667,0.0833333,-1.91667,-0.272727,-0.272727,-0.272727,-0.272727,-0.272727,-0.272727,-0.272727,-0.272727,2.13636,-1.86364,-1.86364,0.136364,-1.86364,-1.86364,2.13636,0.136364,-2.91667,5.08333,6.08333,1.08333,-0.916667,-2.91667,2.08333,-0.916667,-0.416667,-2.91667,5.80682,2.30682,-0.193182,-3.69318,-4.19318,8.80682,-1.69318,-2.19318,4.46667,3.96667,-3.53333,-1.53333,-1.03333,-0.533333,-0.533333,7.46667,-1.03333,-2.03333,3.45833,1.45833,-1.54167,-1.54167,-1.54167,0.458333,0.958333,-1.54167,-1.54167,1.95833,-1.41667,-1.41667,0.583333,0.583333,0.583333,2.58333,1.08333,-1.41667,0.0833333,-1.41667,-1.83333,-1.83333,2.16667,0.166667,0.166667,2.66667,0.166667,3.66667,0.166667,2.85417,-0.145833,4.35417,-1.64583,-1.64583,-1.64583,-1.64583,-1.64583,-1.64583,2.85417,-1.04167,0.458333,-1.04167,2.95833,1.95833,-1.04167,0.458333,-1.04167,-1.04167,1.45833,-1.08333,0.916667,0.916667,-1.08333,-1.08333,0.916667,1.91667,0.916667,-1.08333,-1.08333,5.45833,-1.04167,1.45833,-0.0416667,-1.54167,-1.54167,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,1.375,-0.125,-0.125,-0.125,-3.58333,0.416667,-1.58333,-3.58333,0.416667,-3.58333,-1.58333,0.416667,1.41667,2.91667,-0.125,-0.125,-0.125,1.375,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,-0.375,-0.375,-0.375,1.625,2.125,-0.375,-0.375,-0.375,-0.375,-0.375,1.05556,6.05556,2.05556,6.05556,-3.94444,-2.44444,-1.94444,0.0555556,4.05556,-3.94444,0.125,-1.875,-1.875,1.625,0.125,-1.875,-1.875,-1.875,0.125,0.125,-0.791667,1.20833,-0.791667,-0.791667,0.708333,-0.791667,-0.791667,-0.791667,1.70833,-0.791667)

#' export STAN_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#' time ./bmlm_map_rect sample num_samples=1000 num_warmup=1000 output file=fit_map.csv data file=roi_hlm_input.R

# Sample from model
# message("Estimating model, please wait.")
# fit <- rstan::sampling(
#     object = model_s,
#     data = ld,
#     pars = c("U", "z_U", "V", "z_V", "L_Omega_id", "Tau_id", "Sigma_id", "L_Omega_roi", "Tau_roi", "Sigma_roi"),
#     include = FALSE,
#     chains = 6, iter = 4000, cores = 6, warmup = 1000)
#
# saveRDS(fit, 'data/sea_roi_fit-eptot_gad-td10.rds')
