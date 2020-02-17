// Stan code for multilevel mediation model

// *****
// It is very important that the data be ordered by ROI id!
// *****

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
functions {
    vector hlm_med(vector global, vector Vs,
    real[] xr, int[] xi) {
        int nn = xi[1];
        int P = xi[2];
        int Nr = xi[3];
        int Ni = xi[4];
        int Ly = xi[5];
        int Lm = xi[6];
        int J = xi[7];
        // target += sum(map_rect(hlm_med, append_row(gammas,
        //                                  append_row(ybeta,
        //                                   append_row(mbeta,
        //                                    append_row(to_vector(U), sigmas)))),
        //                        Vs, x_r, x_i));
        vector[P] gammas = global[1:P];
        vector[Ly] ybeta = global[(P+1):(P+Ly)];           // ID-varying covariates to Y
        vector[Lm] mbeta = global[(P+Ly+1):(P+Ly+Lm)];           // ID-varying covariates to M
        matrix[J,P] U = to_matrix(global[(P+Ly+Lm+1):(P+Ly+Lm+J*P)], J, P);
        real sigma_m = global[P+Ly+Lm+J*P+1];      // Residual
        real sigma_y = global[P+Ly+Lm+J*P+2];      // Residual
        // x_r[k, (0*slen+1):(1*slen)] = to_array_1d(Y[ beg:end ]);
        // x_r[k, (1*slen+1):(2*slen)] = to_array_1d(X[ beg:end ]);
        // x_r[k, (2*slen+1):(3*slen)] = to_array_1d(M[ beg:end ]);
        // x_r[k, (3*slen+1):(4*slen)] = to_array_1d(Time[ beg:end ]);
        vector[nn] Y =    to_vector(xr[(0*nn+1):(1*nn)]);
        vector[nn] X =    to_vector(xr[(1*nn+1):(2*nn)]);
        vector[nn] M =    to_vector(xr[(2*nn+1):(3*nn)]);
        vector[nn] Time = to_vector(xr[(3*nn+1):(4*nn)]);
        // //matrix[N, Ly] Cy;
        // for (lyi in 1:Ly){
        //     x_r[k, ((4+lyi-1)*slen+1):((4+lyi)*slen)] = to_array_1d(Cy[ beg:end, lyi ]);
        // }
        // //matrix[N, Lm] Cm;
        // for (lmi in 1:Lm){
        //     x_r[k, ((4+Ly+lmi-1)*slen+1):((4+Ly+lmi)*slen)] = to_array_1d(Cm[ beg:end, lmi ]);
        // }
        matrix[nn, Ly] Cy;
        matrix[nn, Lm] Cm;
        int id[nn] = xi[8:(nn+7)];
        vector[nn] mu_y;
        vector[nn] mu_m;
        real lly;
        real llm;

        for(lyi in 1:Ly){
            Cy[,lyi] = to_vector(xr[((4+lyi-1)*nn+1):((4+lyi)*nn)]);
        }
        for(lmi in 1:Lm){
            Cm[,lmi] = to_vector(xr[((4+Ly+lmi-1)*nn+1):((4+Ly+lmi)*nn)]);
        }
        // Regressions
        //     1 real dy;                    // Intercept
        //     2 real cp;                    // X to Y effect
        //     3 real b;                     // M to Y effect
        //     4 real ty;                    // t to Y effect
        mu_y = (gammas[2] + U[id, 1] + Vs[1]) .* X +
               (gammas[3] + U[id, 2] + Vs[2]) .* M +
               (gammas[4] + U[id, 6] + Vs[6]) .* Time +
               (gammas[1] + Cy*ybeta + U[id, 4] + Vs[4]);
        // Regression M on X
        //     5 real dm;                    // Intercept
        //     6 real a;                     // X to M effect
        //     7 real tm;                    // t to M effect

        mu_m = (gammas[6] + U[id, 3] + Vs[3]) .* X +
               (gammas[7] + U[id, 7] + Vs[7]) .* Time +
               (gammas[5] + Cm*mbeta + U[id, 5] + Vs[5]);
        // // Data model
        // Y ~ normal(mu_y, sigma_y);
        // M ~ normal(mu_m, sigma_m);
        lly = normal_lpdf(Y | mu_y, sigma_y);
        llm = normal_lpdf(M | mu_m, sigma_m);

        return [lly + llm]';
    }
}
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
    int P = 7;                      // Number of person & ROI-varying variables: dm, dy, a, b, cp, ty, tm
                                    //   That is, intercept for m and y equations, a path, b path, c prime path,
                                    //   and 2 time effects.
    int n_int = 7;                   // Extra ints we need for the function
    int Nr = 4 + Ly + Lm;           // Number of real-valued variables: Y, X, M, Time and the covariates
    int Ni = 1;                     // Number of int-valued variables
    int<lower = 0> slen = N / K;    // Number of observations per variable per shard, assumes equal number of obs per K
    real x_r[K, slen * Nr];         // Array with a row per shard, and enough columns to hold all the real data
    int  x_i[K, slen * Ni + n_int]; // Ditto, interger data.

    // Make Shards, one per K rois
    {
        x_i[,1] = rep_array(slen, K);
        x_i[,2] = rep_array(P, K);
        x_i[,3] = rep_array(Nr, K);
        x_i[,4] = rep_array(Ni, K);
        x_i[,5] = rep_array(Ly, K);
        x_i[,6] = rep_array(Lm, K);
        x_i[,7] = rep_array(J, K);
        for (k in 1:K){
            int beg = 1 + (k-1)*slen; //k = 1, beg = 1
            int end = k*slen;         //k = 1, end = slen = N/K
            x_r[k, (0*slen+1):(1*slen)] = to_array_1d(Y[ beg:end ]);
            x_r[k, (1*slen+1):(2*slen)] = to_array_1d(X[ beg:end ]);
            x_r[k, (2*slen+1):(3*slen)] = to_array_1d(M[ beg:end ]);
            x_r[k, (3*slen+1):(4*slen)] = to_array_1d(Time[ beg:end ]);

            //matrix[N, Ly] Cy;
            for (lyi in 1:Ly){
                x_r[k, ((4+lyi-1)*slen+1):((4+lyi)*slen)] = to_array_1d(Cy[ beg:end, lyi ]);
            }
            //matrix[N, Lm] Cm;
            for (lmi in 1:Lm){
                x_r[k, ((4+Ly+lmi-1)*slen+1):((4+Ly+lmi)*slen)] = to_array_1d(Cm[ beg:end, lmi ]);
            }
            //int<lower=1,upper=J> id[N];
            x_i[k, (n_int + 1):(slen * Ni + n_int)] = id[ beg:end ];
        }
    }
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
    // For mapping
    vector[P*(P-1)/2 + P] L_Omega_id_vec;
    vector[P*(P-1)/2 + P] L_Omega_roi_vec;

    //vector[J*P] Us; We can just do this on the fly with `to_vector`
    vector[P] Vs[K];
    vector[2] sigmas;

    sigmas[1] = sigma_m;
    sigmas[2] = sigma_y;

    U = (diag_pre_multiply(Tau_id, L_Omega_id) * z_U)';
    V = (diag_pre_multiply(Tau_roi, L_Omega_roi) * z_V)';

    {
      for (p in 1:P){
        int pm1 = p - 1;
        int start = p + (pm1*(pm1 - 1)/2);
        L_Omega_id_vec[start:(start + pm1)] = to_vector(L_Omega_id[p, 1:p]);
        L_Omega_roi_vec[start:(start + pm1)] = to_vector(L_Omega_roi[p, 1:p]);
      }

      for(k in 1:K){
        Vs[K,] = to_vector(V[K,]);
      }
    }
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

    // // Regressions
    // //     1 real dy;                    // Intercept
    // //     2 real cp;                    // X to Y effect
    // //     3 real b;                     // M to Y effect
    // //     4 real ty;                    // t to Y effect
    // mu_y = (gammas[2] + U[id, 1] + V[roi, 1]) .* X +
    //        (gammas[3] + U[id, 2] + V[roi, 2]) .* M +
    //        (gammas[4] + U[id, 6] + V[roi, 6]) .* Time +
    //        (gammas[1] + Cy*ybeta + U[id, 4] + V[roi, 4]);
    // // Regression M on X
    // //     5 real dm;                    // Intercept
    // //     6 real a;                     // X to M effect
    // //     7 real tm;                    // t to M effect
    //
    // mu_m = (gammas[6] + U[id, 3] + V[roi, 3]) .* X +
    //        (gammas[7] + U[id, 7] + V[roi, 7]) .* Time +
    //        (gammas[5] + Cm*mbeta + U[id, 5] + V[roi, 5]);
    // // Data model
    // Y ~ normal(mu_y, sigma_y);
    // M ~ normal(mu_m, sigma_m);
    if(SIMULATE == 0){
        target += sum(map_rect(hlm_med, append_row(gammas,
                                         append_row(ybeta,
                                          append_row(mbeta,
                                           append_row(to_vector(U), sigmas)))),
                               Vs, x_r, x_i));
    }
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

    for (j in 1:J) {
        u_a[j] = gammas[6] + U[j, 3];
        u_b[j] = gammas[3] + U[j, 2];
        u_cp[j] = gammas[2] + U[j, 1];
        u_dy[j] = gammas[1] + U[j, 4];
        u_dm[j] = gammas[5] + U[j, 5];
        u_ty[j] = gammas[4] + U[j, 6];
        u_tm[j] = gammas[7] + U[j, 7];
        u_me[j] = (gammas[6] + U[j, 3]) * (gammas[3] + U[j, 2]) + covab_roi; // include covariance due to the ROI grouping factor
        u_c[j] = u_cp[j] + u_me[j];
        u_pme[j] = u_me[j] / u_c[j];
    }
    for (k in 1:K) {
        v_a[k] = gammas[6]+ V[k, 3];
        v_b[k] = gammas[3]+ V[k, 2];
        v_cp[k] = gammas[2] + V[k, 1];
        v_dy[k] = gammas[1] + V[k, 4];
        v_dm[k] = gammas[5] + V[k, 5];
        v_ty[k] = gammas[4] + V[k, 6];
        v_tm[k] = gammas[7] + V[k, 7];
        v_me[k] = (gammas[6] + V[k, 3]) * (gammas[3] + V[k, 2]) + covab_id; // inclVde covariance dVe to the ROI groVping factor
        v_c[k] = v_cp[k] + v_me[k];
        v_pme[k] = v_me[k] / v_c[k];
    }

    {
        vector[N] mu_y;
        vector[N] mu_m;

        if(SIMULATE == 1){
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
