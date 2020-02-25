library(cmdstanr)
library(rstan)
set_cmdstan_path('~/code_new/cmdstan/')

datadir <- '/data/jflournoy/SEA/roi_bayes/eptot.gad_lead.fear/roi_hlm/'

stanfit <- rstan::read_stan_csv(paste0('/data/jflournoy/SEA/roi_bayes/eptot.gad_lead.fear/roi_hlm/sea_fit_map_', 1:4, '_brief.csv'))


rstan::summary(stanfit, pars = paste0('v_me[',1:718,']'), probs = c(.025,.975))
bayesplot::mcmc_intervals(stanfit, regex_pars = 'v_me\\[[0-9]{1}\\]')
bayesplot::mcmc_trace(stanfit, regex_pars = 'gammas\\[[123]\\]')
bayesplot::mcmc_trace(stanfit, regex_pars = 'gammas\\[[456]\\]')
bayesplot::mcmc_trace(stanfit, regex_pars = 'v_a\\[[123]\\]')


some_omega <- as.matrix(stanfit, pars = grep('Omega_id.*', names(stanfit), value = T))
some_omega_means <- apply(some_omega, 2, function(acol){
    quantile(acol, probs = c(.025, .5, .975))
})
some_omega_means <- apply(some_omega, 2, function(acol){
    qq <- quantile(acol, probs = c(.025, .5, .975))
    sprintf('%.2f %-.2f %-.2f ', qq[1], qq[2], qq[3])
})
matrix(some_omega_means, ncol = 7)
