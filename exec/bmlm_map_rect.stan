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
    real prior_dm;
    real prior_dy;
    real prior_a;
    real prior_b;
    real prior_cp;
    real prior_ty;
    real prior_tm;
    //ID Priors
    real prior_id_tau_dm;
    real prior_id_tau_dy;
    real prior_id_tau_a;
    real prior_id_tau_b;
    real prior_id_tau_cp;
    real prior_id_tau_ty;
    real prior_id_tau_tm;
    real prior_id_lkj_shape;
    //ROI Priors
    real prior_roi_tau_dm;
    real prior_roi_tau_dy;
    real prior_roi_tau_a;
    real prior_roi_tau_b;
    real prior_roi_tau_cp;
    real prior_roi_tau_ty;
    real prior_roi_tau_tm;
    real prior_roi_lkj_shape;
    //ID varying covars Priors
    real prior_ybeta;
    real prior_mbeta;

    vector[N] Y;                // Continuous outcome
}
transformed data{
    int P = 7;                      // Number of person & ROI-varying variables: dm, dy, a, b, cp, ty, tm
                                    //   That is, intercept for m and y equations, a path, b path, c prime path, and 2 time effects.
    int Nr = 4 + Ly + Lm;           // Number of real-valued variables: Y, X, M, Time and the covariates
    int Ni = 1;                     // Number of int-valued variables
    int<lower = 0> slen = N / K;
    real x_r[K, slen * Nr];         // Array with a row per shard, and enough columns to hold all the real data
    int<lower=1,upper=J> x_i[K, slen * Ni];

    // Make Shards, one per K rois
    {
        for (k in 1:K){
            int beg = 1 + (k-1)*slen;
            int end = k*slen;
            x_r[k, 1:slen] = to_array_1d(Y[ beg:end ]);
            x_r[k, (slen+1):(2*slen)] = to_array_1d(X[ beg:end ]);
            x_r[k, (slen+1):(3*slen)] = to_array_1d(M[ beg:end ]);
            x_r[k, (slen+1):(4*slen)] = to_array_1d(Time[ beg:end ]);

            //matrix[N, Ly] Cy;
            for (lyi in 1:Ly){
                x_r[k, (slen+1):((4 + lyi)*slen)] = to_array_1d(Cy[ beg:end, lyi ]);
            }
            //matrix[N, Lm] Cm;
            for (lmi in 1:Lm){
                x_r[k, (slen+1):((4 + Ly + lmi)*slen)] = to_array_1d(Cm[ beg:end, lmi ]);
            }
            //int<lower=1,upper=J> id[N];
            x_i[k] = id[ beg:end ];
        }
    }
}
parameters{
    // Regression Y on X and M
    real dy;                    // Intercept
    real cp;                    // X to Y effect
    real b;                     // M to Y effect
    real ty;                    // t to Y effect
    vector[Ly] ybeta;           // ID-varying covariates to Y
    // Regression M on X
    real dm;                    // Intercept
    real a;                     // X to M effect
    real tm;                    // t to M effect
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
    dy ~ normal(0, prior_dy);
    dm ~ normal(0, prior_dm);
    ty ~ normal(0, prior_ty);
    tm ~ normal(0, prior_tm);
    a ~ normal(0, prior_a);
    b ~ normal(0, prior_b);
    cp ~ normal(0, prior_cp);
    ybeta ~ normal(0, prior_ybeta); //NOTE: should this be vectorized so prior_ybeta is of length Ly?
    mbeta ~ normal(0, prior_mbeta); //NOTE: ditto above.
    // SDs and correlation matrix ID-varying
    Tau_id[1] ~ cauchy(0, prior_id_tau_cp);   // u_cp
    Tau_id[2] ~ cauchy(0, prior_id_tau_b);    // u_b
    Tau_id[3] ~ cauchy(0, prior_id_tau_a);    // u_a
    Tau_id[4] ~ cauchy(0, prior_id_tau_dy);   // u_intercept_y
    Tau_id[5] ~ cauchy(0, prior_id_tau_dm);   // u_intercept_m
    Tau_id[6] ~ cauchy(0, prior_id_tau_ty);   // u_ty
    Tau_id[7] ~ cauchy(0, prior_id_tau_tm);   // u_tm
    L_Omega_id ~ lkj_corr_cholesky(prior_id_lkj_shape);
    // SDs and correlation matrix ROI-varying
    Tau_roi[1] ~ cauchy(0, prior_roi_tau_cp);   // v_cp
    Tau_roi[2] ~ cauchy(0, prior_roi_tau_b);    // v_b
    Tau_roi[3] ~ cauchy(0, prior_roi_tau_a);    // v_a
    Tau_roi[4] ~ cauchy(0, prior_roi_tau_dy);   // v_intercept_y
    Tau_roi[5] ~ cauchy(0, prior_roi_tau_dm);   // v_intercept_m
    Tau_roi[6] ~ cauchy(0, prior_roi_tau_ty);   // v_ty
    Tau_roi[7] ~ cauchy(0, prior_roi_tau_tm);   // v_tm
    L_Omega_roi ~ lkj_corr_cholesky(prior_roi_lkj_shape);

    // Allow vectorized sampling of varying effects via stdzd z_U, z_V
    to_vector(z_U) ~ normal(0, 1);
    to_vector(z_V) ~ normal(0, 1);//shardable over K rois

    // Regressions
    for (n in 1:N){
        mu_y[n] = (cp + U[id[n], 1] + V[roi[n], 1]) * X[n] +
                  (b + U[id[n], 2] + V[roi[n], 2]) * M[n] +
                  (ty + U[id[n], 6] + V[roi[n], 6]) * Time[n] +
                  (dy + Cy[n]*ybeta + U[id[n], 4] + V[roi[n], 4]);
        mu_m[n] = (a + U[id[n], 3] + V[roi[n], 3]) * X[n] +
                  (tm + U[id[n], 7] + V[roi[n], 7]) * Time[n] +
                  (dm + Cm[n]*mbeta + U[id[n], 5] + V[roi[n], 5]);
    }
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
    me = a*b + covab_id + covab_roi;
    c = cp + me;
    pme = me / c;

    for (j in 1:J) {
        u_a[j] = a + U[j, 3];
        u_b[j] = b + U[j, 2];
        u_me[j] = (a + U[j, 3]) * (b + U[j, 2]) + covab_roi; // include covariance due to the ROI grouping factor
        u_cp[j] = cp + U[j, 1];
        u_dy[j] = dy + U[j, 4];
        u_dm[j] = dm + U[j, 5];
        u_ty[j] = ty + U[j, 6];
        u_tm[j] = tm + U[j, 7];
        u_c[j] = u_cp[j] + u_me[j];
        u_pme[j] = u_me[j] / u_c[j];
    }
    for (k in 1:K) {
        v_a[k] = a + V[k, 3];
        v_b[k] = b + V[k, 2];
        v_me[k] = (a + V[k, 3]) * (b + V[k, 2]) + covab_id; // inclVde covariance dVe to the ROI groVping factor
        v_cp[k] = cp + V[k, 1];
        v_dy[k] = dy + V[k, 4];
        v_dm[k] = dm + V[k, 5];
        v_ty[k] = ty + V[k, 6];
        v_tm[k] = tm + V[k, 7];
        v_c[k] = v_cp[k] + v_me[k];
        v_pme[k] = v_me[k] / v_c[k];
    }
}
