library(cmdstanr)
library(rstan)
library(bayesplot)

set_cmdstan_path('~/code_new/cmdstan/')
datadir <- '/data/jflournoy/SEA/roi_bayes/eptot.gad_lead.fear/'



stanfit <- rstan::read_stan_csv(paste0(datadir, 'sea_fit_map_td15_', 1:4, '_brief.csv'))

orderd_rx <- function(afit, terms, indices){
    pars_ <- expand.grid(index = indices, term = terms)
    pars <- paste0(pars_[, 'term'], '[', pars_[, 'index'], ']')
    sum <- rstan::summary(afit, pars = pars, probs = c(.5))
    pars_ordered_ <- dimnames(sum$summary)[[1]][order(sum$summary[,'mean'])]
    pars_ordered <- lapply(terms, function(term){
        grep(term, pars_ordered_, value = T)
    })
    return(pars_ordered)
}

roi_pars_ordered <- orderd_rx(stanfit,
                              terms = list(a = 'v_a', b = 'v_b',
                                           cp = 'v_cp',
                                           me = 'v_me', pme = 'v_pme'),
                              indices = 1:718)

id_pars_ordered <- orderd_rx(stanfit,
                             terms = list(a = 'u_a', b = 'u_b', me = 'u_me'),
                             indices = 1:29)

color_scheme_set("red")
roi_a_plot <- bayesplot::mcmc_intervals(stanfit, pars = roi_pars_ordered$a,
                                        point_est = 'mean', prob = .1, prob_outer = .95)
roi_a_plot$layers[[3]]$aes_params$size <- .1
roi_a_plot$layers[[2]]$aes_params$size <- .1
roi_a_plot$layers[[4]]$aes_params$size <- 1

bayesplot::mcmc_intervals(stanfit, pars = roi_pars_ordered$b)
bayesplot::mcmc_intervals(stanfit, pars = roi_pars_ordered$me)

bayesplot::mcmc_intervals(stanfit, pars = id_pars_ordered$me)
bayesplot::mcmc_intervals(stanfit, pars = id_pars_ordered$a)
bayesplot::mcmc_intervals(stanfit, pars = id_pars_ordered$b)
bayesplot::mcmc_trace(stanfit, regex_pars = 'gammas\\[[123]\\]')
bayesplot::mcmc_trace(stanfit, regex_pars = 'gammas\\[[456]\\]')
bayesplot::mcmc_trace(stanfit, regex_pars = 'v_a\\[[123]\\]')
bayesplot::mcmc_trace(stanfit, regex_pars = 'u_a\\[[123]\\]')
bayesplot::mcmc_trace(stanfit, regex_pars = 'Tau_id\\[[1-6]\\]')
bayesplot::mcmc_trace(stanfit, regex_pars = 'Tau_roi\\[[1-6]\\]')


