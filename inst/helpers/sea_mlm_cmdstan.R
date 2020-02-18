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
y = "GAD7_TOT"
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
    mbeta = 1000
)
if (is.null(priors$bs)) priors$bs <- default_priors$bs
if (is.null(priors$id_taus)) priors$id_taus <- default_priors$id_taus
if (is.null(priors$id_lkj_shape)) priors$id_lkj_shape <- default_priors$id_lkj_shape
if (is.null(priors$roi_taus)) priors$roi_taus <- default_priors$roi_taus
if (is.null(priors$roi_lkj_shape)) priors$roi_lkj_shape <- default_priors$roi_lkj_shape
if (is.null(priors$mbeta)) priors$mbeta <- default_priors$mbeta
if (is.null(priors$ybeta)) priors$ybeta <- default_priors$ybeta
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

#funcs <- rstan::expose_stan_functions('exec/bmlm_map_rect.stan')

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
