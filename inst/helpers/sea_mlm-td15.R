library(data.table)
library(rstan)
setwd('~/otherhome/code/bmlm/')

model_s <- rstan::stan_model('exec/bmlm.stan', model_name = 'mlm', auto_write = TRUE)

load('data/sea_fearGTcalm_stress_psych.rda')

sea_fearGTcalm_stress_psych[, id_idx := as.numeric(as.factor(sea_fearGTcalm_stress_psych$idnum))]
sea_fearGTcalm_stress_psych[, roi_idx := as.numeric(as.factor(sea_fearGTcalm_stress_psych$roi))]
names(sea_fearGTcalm_stress_psych)

#for testing, select random set of ROIs
# set.seed(2242211)
# test_rois <- sample(sea_fearGTcalm_stress_psych[, unique(roi_idx)], 10)

###

# x -> m
# fearGTcalm_WCEN ~ 1 + TIMECENTER + WCEN_EPISODICTOT + GCEN_EPISODICTOT
#
# x, m -> y
# GAD7_TOT ~ 1 + TIMECENTER + WCEN_EPISODICTOT + GCEN_EPISODICTOT + fearGTcalm_WCEN + fearGTcalm_GCEN

# d = copy(sea_fearGTcalm_stress_psych[roi_idx %in% test_rois])
d = copy(sea_fearGTcalm_stress_psych)
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
    dm = 1000, id_tau_dm = 50, roi_tau_dm = 50,
    dy = 1000, id_tau_dy = 50, roi_tau_dy = 50,
    a = 1000, id_tau_a = 50, roi_tau_a = 50,
    b = 1000, id_tau_b = 50, roi_tau_b = 50,
    cp = 1000, id_tau_cp = 50, roi_tau_cp = 50,
    ty = 1000, id_tau_ty = 50, roi_tau_ty = 50,
    tm = 1000, id_tau_tm = 50, roi_tau_tm = 50,
    id_lkj_shape = 1, roi_lkj_shape = 1,
    ybeta = 1000,
    mbeta = 1000
)
if (is.null(priors$dm)) priors$dm <- default_priors$dm
if (is.null(priors$dy)) priors$dy <- default_priors$dy
if (is.null(priors$a)) priors$a <- default_priors$a
if (is.null(priors$b)) priors$b <- default_priors$b
if (is.null(priors$cp)) priors$cp <- default_priors$cp
if (is.null(priors$tm)) priors$tm <- default_priors$tm
if (is.null(priors$ty)) priors$ty <- default_priors$ty
if (is.null(priors$id_tau_dm)) priors$id_tau_dm <- default_priors$id_tau_dm
if (is.null(priors$id_tau_dy)) priors$id_tau_dy <- default_priors$id_tau_dy
if (is.null(priors$id_tau_a)) priors$id_tau_a <- default_priors$id_tau_a
if (is.null(priors$id_tau_b)) priors$id_tau_b <- default_priors$id_tau_b
if (is.null(priors$id_tau_cp)) priors$id_tau_cp <- default_priors$id_tau_cp
if (is.null(priors$id_tau_tm)) priors$id_tau_tm <- default_priors$id_tau_tm
if (is.null(priors$id_tau_ty)) priors$id_tau_ty <- default_priors$id_tau_ty
if (is.null(priors$id_lkj_shape)) priors$id_lkj_shape <- default_priors$id_lkj_shape
if (is.null(priors$roi_tau_dm)) priors$roi_tau_dm <- default_priors$roi_tau_dm
if (is.null(priors$roi_tau_dy)) priors$roi_tau_dy <- default_priors$roi_tau_dy
if (is.null(priors$roi_tau_a)) priors$roi_tau_a <- default_priors$roi_tau_a
if (is.null(priors$roi_tau_b)) priors$roi_tau_b <- default_priors$roi_tau_b
if (is.null(priors$roi_tau_cp)) priors$roi_tau_cp <- default_priors$roi_tau_cp
if (is.null(priors$roi_tau_tm)) priors$roi_tau_tm <- default_priors$roi_tau_tm
if (is.null(priors$roi_tau_ty)) priors$roi_tau_ty <- default_priors$roi_tau_ty
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

# Sample from model
message("Estimating model, please wait.")
fit <- rstan::sampling(
    object = model_s,
    data = ld,
    pars = c("U", "z_U", "V", "z_V", "L_Omega_id", "Tau_id", "Sigma_id", "L_Omega_roi", "Tau_roi", "Sigma_roi"),
    include = FALSE,
    control = list(max_treedepth = 15),
    chains = 6, iter = 4000, cores = 6, warmup = 1000)

saveRDS(fit, 'data/sea_roi_fit-eptot_gad-td15.rds')
