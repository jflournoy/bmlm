library(data.table)

roi <- readRDS('~/otherhome/code/bmlm/data/SEA_face_reactivity_ColeAnticevic_fear.rds')
psych <- readr::read_csv('~/NewNeuropoint/datafiles/stress_psych-midpoint5.csv')

sea_fearGTcalm_stress_psych <- data.table::merge.data.table(roi, psych, by = c('idnum', 'time'), all.x = TRUE)

#Calculate group mean across all observations for each ROI to use to center the
#trait level ROI activity:
sea_fearGTcalm_stress_psych[, fGTc_roi_mean := mean(fearGTcalm, na.rm = T), by = .(roi)]
#Compute the within-person, within-ROI, wave-to-wave fluctation of activity, and
#center the trait-level ROI activity:
sea_fearGTcalm_stress_psych[, c('fGTc_id_roi_mean',
                                'fearGTcalm_WCEN',
                                'fearGTcalm_GCEN') :=
                                .(mean(fearGTcalm), #within person, within roi mean across all waves
                                  fearGTcalm - mean(fearGTcalm), #wave-to-wave fluctuations
                                  mean(fearGTcalm) - fGTc_roi_mean), #center the trait-level activity at group mean for roi
                            by = .(idnum, roi)]

format(object.size(sea_fearGTcalm_stress_psych), units = 'Mb')
usethis::use_data(sea_fearGTcalm_stress_psych, overwrite = TRUE)
