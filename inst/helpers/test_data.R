library(dplyr)
library(tidyr)
library(rstan)

DF1 <- readr::read_csv('~/Downloads/df_SEA_ROIs(1).csv')
DF2 <- readr::read_csv('~/Downloads/stresscov-midpoint5.csv')
DF3 <- readr::read_csv('~/Downloads/depanxcov-midpoint5.csv')

sea_data_ <- dplyr::left_join(dplyr::left_join(DF1, DF2, by = c('ID' = 'ID', 'Month.Number' = 'TIME')), DF3, by = c('ID' = 'idnum', 'Month.Number' = 'time', 'TIMECENTER' = 'TIMECENTER')) %>%
    select(-matches('.*past$'), -matches('.*_with_errors$'))

sea_data <- gather(sea_data_, roi, brain, matches('[LR]_')) %>%
    tidyr::extract(col = roi, into = c('roi', 'contrast', 'vartype'),
                   '((?:[LR]|Bilateral)_.*)_((?:Calm|Fear)_GT_(?:Calm|Scramble))(.gmc|.pmc)*') %>%
    mutate(vartype = case_when(is.na(vartype) ~ 'RAW',
                               vartype == '.gmc' ~ 'GCEN',
                               vartype == '.pmc' ~ 'WCEN')) %>%
    unite(contrast_type, contrast, vartype) %>%
    spread(contrast_type, brain) %>%
    mutate(id_idx = as.numeric(factor(ID, levels = unique(ID))),
           roi_idx = as.numeric(factor(roi, levels = unique(roi)))) %>%
    select(id_idx, roi_idx, WCEN_EPISODICTOT,
           Fear_GT_Calm_WCEN, GAD7_TOT, TIMECENTER,
           GCEN_EPISODICTOT, Fear_GT_Calm_GCEN)



usethis::use_data(sea_data, overwrite = TRUE)
