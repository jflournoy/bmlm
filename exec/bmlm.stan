// Stan code for multilevel mediation model

data {
    int<lower=1> N;             // Number of observations
    int<lower=1> J;             // Number of participants
    int<lower=1,upper=J> id[N]; // Participant IDs
    vector[N] X;                // Manipulated variable
    vector[N] M;                // Mediator
    // Priors
    real prior_dm;
    real prior_dy;
    real prior_a;
    real prior_b;
    real prior_cp;
    real prior_tau_dm;
    real prior_tau_dy;
    real prior_tau_a;
    real prior_tau_b;
    real prior_tau_cp;
    real prior_lkj_shape;
    vector[N] Y;                // Continuous outcome
}
transformed data{
    int K;                      // Number of predictors
    K = 5;
}
parameters{
    // Regression Y on X and M
    real dy;                    // Intercept
    real cp;                    // X to Y effect
    real b;                     // M to Y effect
    // Regression M on X
    real dm;                    // Intercept
    real a;                     // X to M effect
    real<lower=0> sigma_m;      // Residual

    // Correlation matrix and SDs of participant-level varying effects
    cholesky_factor_corr[K] L_Omega;
    vector<lower=0>[K] Tau;

    // Standardized varying effects
    matrix[K, J] z_U;
    real<lower=0> sigma_y;      // Residual
}
transformed parameters {
    // Participant-level varying effects
    matrix[J, K] U;
    U = (diag_pre_multiply(Tau, L_Omega) * z_U)';
}
model {
    // Means of linear models
    vector[N] mu_y;
    vector[N] mu_m;
    // Regression parameter priors
    dy ~ normal(0, prior_dy);
    dm ~ normal(0, prior_dm);
    a ~ normal(0, prior_a);
    b ~ normal(0, prior_b);
    cp ~ normal(0, prior_cp);
    // SDs and correlation matrix
    Tau[1] ~ cauchy(0, prior_tau_cp);   // u_cp
    Tau[2] ~ cauchy(0, prior_tau_b);    // u_b
    Tau[3] ~ cauchy(0, prior_tau_a);    // u_a
    Tau[4] ~ cauchy(0, prior_tau_dy);   // u_intercept_y
    Tau[5] ~ cauchy(0, prior_tau_dm);   // u_intercept_m
    L_Omega ~ lkj_corr_cholesky(prior_lkj_shape);
    // Allow vectorized sampling of varying effects via stdzd z_U
    to_vector(z_U) ~ normal(0, 1);

    // Regressions
    for (n in 1:N){
        mu_y[n] = (cp + U[id[n], 1]) * X[n] +
                  (b + U[id[n], 2]) * M[n] +
                  (dy + U[id[n], 4]);
        mu_m[n] = (a + U[id[n], 3]) * X[n] +
                  (dm + U[id[n], 5]);
    }
    // Data model
    Y ~ normal(mu_y, sigma_y);
    M ~ normal(mu_m, sigma_m);
}
generated quantities{
    matrix[K, K] Omega;         // Correlation matrix
    matrix[K, K] Sigma;         // Covariance matrix

    // Average mediation parameters
    real covab;                 // a-b covariance
    real corrab;                // a-b correlation
    real me;                    // Mediated effect
    real c;                     // Total effect
    real pme;                   // % mediated effect

    // Person-specific mediation parameters
    vector[J] u_a;
    vector[J] u_b;
    vector[J] u_cp;
    vector[J] u_dy;
    vector[J] u_dm;
    vector[J] u_c;
    vector[J] u_me;
    vector[J] u_pme;

    // Re-named tau parameters for easy output
    real tau_cp;
    real tau_b;
    real tau_a;
    real tau_dy;
    real tau_dm;

    tau_cp = Tau[1];
    tau_b = Tau[2];
    tau_a = Tau[3];
    tau_dy = Tau[4];
    tau_dm = Tau[5];

    Omega = L_Omega * L_Omega';
    Sigma = quad_form_diag(Omega, Tau);

    covab = Sigma[3,2];
    corrab = Omega[3,2];
    me = a*b + covab;
    c = cp + me;
    pme = me / c;

    for (j in 1:J) {
        u_a[j] = a + U[j, 3];
        u_b[j] = b + U[j, 2];
        u_me[j] = (a + U[j, 3]) * (b + U[j, 2]);
        u_cp[j] = cp + U[j, 1];
        u_dy[j] = dy + U[j, 4];
        u_dm[j] = dm + U[j, 5];
        u_c[j] = u_cp[j] + u_me[j];
        u_pme[j] = u_me[j] / u_c[j];
    }
}
