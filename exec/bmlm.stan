// Stan code for multilevel mediation model

// Translating the following:
// PSYCH_SYMPTOM ~ 1 + TIMECENTER + WCEN_EPISODIC + GCEN_EPISODIC + WCEN_BRAIN + GCEN_BRAIN
// BRAIN ~ 1 + TIMECENTER + WCEN_EPISODIC + GCEN_EPISODIC
//
// where y is PSYCH_SYMPTOM, X is WCEN_EPISODIC, and M is WCEN_BRAIN.
//
// We aim to allow mediated effects to vary across person and across ROI.
//
// Y Equation
//   level 1:
//     y ~ dy_ + ty_*TIME + cp_*X + b_*M
//   level 2:
//     dy_ ~ dy + Cy*ybeta + U[id, 4] + V[roi, 4]
//     ty_ ~ ty + U[id, 6] + V[roi, 6]
//     cp_ ~ cp + U[id, 1] + V[roi, 1]
//     b_ ~ b + U[id, 2] + V[roi, 2]
//
// M Equation
//   level 1:
//     m ~ dm_ + tm_*TIME + a_*X
//   level 2:
//     dm_ ~ dm + Cm*mbeta + U[id, 5] + V[roi, 5]
//     tm_ ~ tm + U[id, 7] + V[roi, 7]
//     a_ ~ a + U[id, 3] + V[roi, 3]
//
// In the above, Cy is a matrix containing scores for both GCEN_EPISODIC and GCEN_BRAIN,
// while Cm is a matrix containing scores for GCEN_EPISODIC.

data {
    int<lower=1> N;             // Number of observations
    int<lower=1> J;             // Number of participants
    int<lower=1> K;             // Number of ROIs
    int<lower=1> Ly;            // Number of participant-varying covariates, y equation
    int<lower=1> Lm;            // Number of participant-varying covariates, m equation
    int<lower=1,upper=J> id[N]; // Participant IDs
    int<lower=1,upper=K> roi[N];// ROI ids
    vector[N] X;                // Treatment variable
    vector[N] M;                // Mediator
    vector[N] Time;             // Time variable for de-meaning
    matrix[N, Ly] Cy;           // participant/ROI-varying covariates, y equation - we just assume the coefficients for these do not vary by ID or ROI though the values of the variables might differ within participant by ROI, or within ROI by participant
    matrix[N, Lm] Cm;           // participant/ROI-varying covariates, m equation - we just assume the coefficients for these do not vary by ID or ROI though the values of the variables might differ within participant by ROI, or within ROI by participant
    //// Values here allow user to specificy different SDs for the normal
    //// (centered at 0) priors for all these parameters
    // Population Param Priors
    real prior_bs;
    //ID Priors
    real prior_id_taus;
    real prior_id_lkj_shape;
    //ROI Priors
    real prior_roi_taus;
    real prior_roi_lkj_shape;
    //ID varying covars Priors
    real prior_ybeta;
    real prior_mbeta;

    int<lower=0,upper=1> SIMULATE; //should we just simulate values?
    vector[N] Y;                // Continuous outcome
}
transformed data{
    int P;                      // Number of person & ROI-varying variables
    P = 7;                      // dm, dy, a, b, cp, ty, tm
}
parameters{
    // Regression Y on X and M
    vector[P] gammas;
    //     1 real dy;                    // Intercept
    //     2 real cp;                    // X to Y effect
    //     3 real b;                     // M to Y effect
    //     4 real ty;                    // t to Y effect
    // Regression M on X
    //     5 real dm;                    // Intercept
    //     6 real a;                     // X to M effect
    //     7 real tm;                    // t to M effect
    vector[Ly] ybeta;           // ID-varying covariates to Y
    vector[Lm] mbeta;           // ID-varying covariates to M
    real<lower=0> sigma_m;      // Residual

    // Correlation matrix and SDs of participant-level varying effects
    cholesky_factor_corr[P] L_Omega_id;
    vector<lower=0>[P] Tau_id;
    // Correlation matrix and SDs of roi-level varying effects
    cholesky_factor_corr[P] L_Omega_roi;
    vector<lower=0>[P] Tau_roi;

    // Standardized varying effects
    matrix[P, J] z_U;
    matrix[P, K] z_V;           //shardable over K ROIS
    real<lower=0> sigma_y;      // Residual
}
transformed parameters {
    // Participant-level varying effects
    matrix[J, P] U;
    // ROI-level varying effects
    matrix[K, P] V;
    U = (diag_pre_multiply(Tau_id, L_Omega_id) * z_U)';
    V = (diag_pre_multiply(Tau_roi, L_Omega_roi) * z_V)';
}
model {
    // Means of linear models
    vector[N] mu_y;
    vector[N] mu_m;
    // Regression parameter priors
    gammas ~ normal(0, prior_bs);
    ybeta ~ normal(0, prior_ybeta);
    mbeta ~ normal(0, prior_mbeta);
    // SDs and correlation matrix ID-varying
    Tau_id ~ cauchy(0, prior_id_taus);        // 1 u_cp
                                              // 2 u_b
                                              // 3 u_a
                                              // 4 u_intercept_y
                                              // 5 u_intercept_m
                                              // 6 u_ty
                                              // 7 u_tm
    L_Omega_id ~ lkj_corr_cholesky(prior_id_lkj_shape);
    // SDs and correlation matrix ROI-varying
    Tau_roi ~ cauchy(0, prior_roi_taus);      // 1 u_cp
                                              // 2 u_b
                                              // 3 u_a
                                              // 4 u_intercept_y
                                              // 5 u_intercept_m
                                              // 6 u_ty
                                              // 7 u_tm
    L_Omega_roi ~ lkj_corr_cholesky(prior_roi_lkj_shape);

    // Allow vectorized sampling of varying effects via stdzd z_U, z_V
    to_vector(z_U) ~ normal(0, 1);
    to_vector(z_V) ~ normal(0, 1);//shardable over K rois
    // Regressions
    //     1 real dy;                    // Intercept
    //     2 real cp;                    // X to Y effect
    //     3 real b;                     // M to Y effect
    //     4 real ty;                    // t to Y effect
    // tau_roi_cp = Tau_roi[1];
    // tau_roi_b = Tau_roi[2];
    // tau_roi_a = Tau_roi[3];
    // tau_roi_dy = Tau_roi[4];
    // tau_roi_dm = Tau_roi[5];
    // tau_roi_ty = Tau_roi[6];
    // tau_roi_tm = Tau_roi[7];
    mu_y = (gammas[2] + U[id, 1] + V[roi, 1]) .* X +       // cp
           (gammas[3] + U[id, 2] + V[roi, 2]) .* M +       // b
           (gammas[4] + U[id, 6] + V[roi, 6]) .* Time +    // ty
           (gammas[1] + Cy*ybeta + U[id, 4] + V[roi, 4]);  // dy
    // Regression M on X
    //     5 real dm;                    // Intercept
    //     6 real a;                     // X to M effect
    //     7 real tm;                    // t to M effect

    mu_m = (gammas[6] + U[id, 3] + V[roi, 3]) .* X +       // a
           (gammas[7] + U[id, 7] + V[roi, 7]) .* Time +    // tm
           (gammas[5] + Cm*mbeta + U[id, 5] + V[roi, 5]);  // dm
    // Data model
    Y ~ normal(mu_y, sigma_y);
    M ~ normal(mu_m, sigma_m);
}
generated quantities{
    //NOTE: Include relevant generated quantities for new ROI-varying effect covariance
    matrix[P, P] Omega_id;         // Correlation matrix
    matrix[P, P] Sigma_id;         // Covariance matrix
    matrix[P, P] Omega_roi;         // Correlation matrix
    matrix[P, P] Sigma_roi;         // Covariance matrix

    // Average mediation parameters
    real covab_id;              // a-b covariance across IDs
    real corrab_id;             // a-b correlation across IDs
    real covab_roi;             // a-b covariance acrosss ROIs
    real corrab_roi;            // a-b correlation acrosss ROIs
    real me;                    // Mediated effect
    real c;                     // Total effect
    real pme;                   // % mediated effect

    // Person-specific mediation parameters
    vector[J] u_a;
    vector[J] u_b;
    vector[J] u_cp;
    vector[J] u_dy;
    vector[J] u_dm;
    vector[J] u_ty;
    vector[J] u_tm;
    vector[J] u_c;
    vector[J] u_me;
    vector[J] u_pme;
    // ROI-specific mediation parameters
    vector[K] v_a;
    vector[K] v_b;
    vector[K] v_cp;
    vector[K] v_dy;
    vector[K] v_dm;
    vector[K] v_ty;
    vector[K] v_tm;
    vector[K] v_c;
    vector[K] v_me;
    vector[K] v_pme;

    // Re-named tau parameters for easy output
    real tau_id_cp;
    real tau_id_b;
    real tau_id_a;
    real tau_id_dy;
    real tau_id_dm;
    real tau_id_ty;
    real tau_id_tm;
    real tau_roi_cp;
    real tau_roi_b;
    real tau_roi_a;
    real tau_roi_dy;
    real tau_roi_dm;
    real tau_roi_ty;
    real tau_roi_tm;

    real Y_sim[N];
    real M_sim[N];

    tau_id_cp = Tau_id[1];
    tau_id_b = Tau_id[2];
    tau_id_a = Tau_id[3];
    tau_id_dy = Tau_id[4];
    tau_id_dm = Tau_id[5];
    tau_id_ty = Tau_id[6];
    tau_id_tm = Tau_id[7];
    tau_roi_cp = Tau_roi[1];
    tau_roi_b = Tau_roi[2];
    tau_roi_a = Tau_roi[3];
    tau_roi_dy = Tau_roi[4];
    tau_roi_dm = Tau_roi[5];
    tau_roi_ty = Tau_roi[6];
    tau_roi_tm = Tau_roi[7];

    Omega_id = L_Omega_id * L_Omega_id';
    Sigma_id = quad_form_diag(Omega_id, Tau_id);
    Omega_roi = L_Omega_roi * L_Omega_roi';
    Sigma_roi = quad_form_diag(Omega_roi, Tau_roi);

    //NOTE: We need to figure out what is the proper way to
    //      acount for covariance between a and b paths
    //      across both grouping factors (ID and ROI,
    //      crossed, not nested).
    //
    //      I've taken a stab at something that might
    //      be correct but it's a very naive extension
    //      of the case where there is only one grouping
    //      factor.
    covab_id = Sigma_id[3,2];
    corrab_id = Omega_id[3,2];
    covab_roi = Sigma_roi[3,2];
    corrab_roi = Omega_roi[3,2];
    // vector[P] gammas;
    // //     1 real dy;                    // Intercept
    // //     2 real cp;                    // X to Y effect
    // //     3 real b;                     // M to Y effect
    // //     4 real ty;                    // t to Y effect
    // // Regression M on X
    // //     5 real dm;                    // Intercept
    // //     6 real a;                     // X to M effect
    // //     7 real tm;                    // t to M effect
    me = gammas[6]*gammas[3] + covab_id + covab_roi;
    c = gammas[2] + me;
    pme = me / c;

    u_a = gammas[6] + U[, 3];
    u_b = gammas[3] + U[, 2];
    u_cp = gammas[2] + U[, 1];
    u_dy = gammas[1] + U[, 4];
    u_dm = gammas[5] + U[, 5];
    u_ty = gammas[4] + U[, 6];
    u_tm = gammas[7] + U[, 7];
    u_me = (gammas[6] + U[, 3]) .* (gammas[3] + U[, 2]) + covab_roi; // include covariance due to the ROI grouping factor
    u_c = u_cp + u_me;
    u_pme = u_me ./ u_c;

    v_a = gammas[6]+ V[, 3];
    v_b = gammas[3]+ V[, 2];
    v_cp = gammas[2] + V[, 1];
    v_dy = gammas[1] + V[, 4];
    v_dm = gammas[5] + V[, 5];
    v_ty = gammas[4] + V[, 6];
    v_tm = gammas[7] + V[, 7];
    v_me = (gammas[6] + V[, 3]) .* (gammas[3] + V[, 2]) + covab_id; // include covariance due to the ROI grouping factor
    v_c = v_cp + v_me;
    v_pme = v_me ./ v_c;

    {
        if(SIMULATE == 1){
            vector[N] mu_y;
            vector[N] mu_m;
            // Regressions
            //     1 real dy;                    // Intercept
            //     2 real cp;                    // X to Y effect
            //     3 real b;                     // M to Y effect
            //     4 real ty;                    // t to Y effect
            mu_y = (gammas[2] + U[id, 1] + V[roi, 1]) .* X +
                   (gammas[3] + U[id, 2] + V[roi, 2]) .* M +
                   (gammas[4] + U[id, 6] + V[roi, 6]) .* Time +
                   (gammas[1] + Cy*ybeta + U[id, 4] + V[roi, 4]);
            // Regression M on X
            //     5 real dm;                    // Intercept
            //     6 real a;                     // X to M effect
            //     7 real tm;                    // t to M effect

            mu_m = (gammas[6] + U[id, 3] + V[roi, 3]) .* X +
                   (gammas[7] + U[id, 7] + V[roi, 7]) .* Time +
                   (gammas[5] + Cm*mbeta + U[id, 5] + V[roi, 5]);
            // Data model
            Y_sim = normal_rng(mu_y, sigma_y);
            M_sim = normal_rng(mu_m, sigma_m);
        }
    }
}
