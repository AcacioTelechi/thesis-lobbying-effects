
options(warn=1)
library(fixest); 
library(jsonlite)
df <- read.csv('C:/Users/caca_/AppData/Local/Temp/tmpbtod1qtp/ppml_input.csv', stringsAsFactors=FALSE)
# Ensure proper types
df$questions <- as.numeric(df$questions)
df$meetings <- as.numeric(df$meetings)
df$domain <- factor(df$domain)
if ("agriculture" %in% levels(df$domain)) {
  df$domain <- relevel(df$domain, ref = "agriculture")
}

                fml <- as.formula(questions ~ meetings + i(domain, meetings) + meps_POLITICAL_GROUP_5148.0 + meps_POLITICAL_GROUP_5151.0 + meps_POLITICAL_GROUP_5152.0 + meps_POLITICAL_GROUP_5153.0 + meps_POLITICAL_GROUP_5154.0 + meps_POLITICAL_GROUP_5155.0 + meps_POLITICAL_GROUP_5588.0 + meps_POLITICAL_GROUP_5704.0 + meps_POLITICAL_GROUP_6259.0 + meps_POLITICAL_GROUP_6561.0 + meps_POLITICAL_GROUP_7018.0 + meps_POLITICAL_GROUP_7028.0 + meps_POLITICAL_GROUP_7035.0 + meps_POLITICAL_GROUP_7036.0 + meps_POLITICAL_GROUP_7037.0 + meps_POLITICAL_GROUP_7038.0 + meps_POLITICAL_GROUP_7150.0 + meps_POLITICAL_GROUP_7151.0 + meps_COUNTRY_AUT + meps_COUNTRY_BEL + meps_COUNTRY_BGR + meps_COUNTRY_CYP + meps_COUNTRY_CZE + meps_COUNTRY_DEU + meps_COUNTRY_DNK + meps_COUNTRY_ESP + meps_COUNTRY_EST + meps_COUNTRY_FIN + meps_COUNTRY_FRA + meps_COUNTRY_GBR + meps_COUNTRY_GRC + meps_COUNTRY_HRV + meps_COUNTRY_HUN + meps_COUNTRY_IRL + meps_COUNTRY_ITA + meps_COUNTRY_LTU + meps_COUNTRY_LUX + meps_COUNTRY_LVA + meps_COUNTRY_MLT + meps_COUNTRY_NLD + meps_COUNTRY_POL + meps_COUNTRY_PRT + meps_COUNTRY_ROU + meps_COUNTRY_SVK + meps_COUNTRY_SVN + meps_COUNTRY_SWE + meps_COMMITTEE_PARLIAMENTARY_SPECIAL___CHAIR + meps_COMMITTEE_PARLIAMENTARY_SPECIAL___MEMBER + meps_COMMITTEE_PARLIAMENTARY_STANDING___CHAIR + meps_COMMITTEE_PARLIAMENTARY_STANDING___MEMBER + meps_COMMITTEE_PARLIAMENTARY_SUB___CHAIR + meps_COMMITTEE_PARLIAMENTARY_SUB___MEMBER + meps_DELEGATION_PARLIAMENTARY___CHAIR + meps_DELEGATION_PARLIAMENTARY___MEMBER + meps_EU_INSTITUTION___PRESIDENT + meps_EU_INSTITUTION___QUAESTOR + meps_EU_POLITICAL_GROUP___CHAIR + meps_EU_POLITICAL_GROUP___MEMBER_BUREAU + meps_EU_POLITICAL_GROUP___TREASURER + meps_EU_POLITICAL_GROUP___TREASURER_CO + meps_NATIONAL_CHAMBER___PRESIDENT_VICE + meps_WORKING_GROUP___CHAIR + meps_WORKING_GROUP___MEMBER + meps_WORKING_GROUP___MEMBER_BUREAU + log_meetings_member_capacity_Committee_chair + log_meetings_member_capacity_Delegation_chair + log_meetings_member_capacity_Member + log_meetings_member_capacity_Rapporteur + log_meetings_member_capacity_Rapporteur_for_opinion + log_meetings_member_capacity_Shadow_rapporteur + log_meetings_member_capacity_Shadow_rapporteur_for_opinion | member_id + time_fe)
cl  <- as.formula('~ member_id')
fit <- tryCatch(fepois(fml, data=df, cluster=cl), error=function(e) e)
if (inherits(fit, 'error')) {
  write(toJSON(list(error=fit$message), auto_unbox=TRUE), file='C:/Users/caca_/AppData/Local/Temp/tmpbtod1qtp/ppml_output.json'); quit(status=0)
}
sm <- summary(fit)
cv <- tryCatch(fit$collin.var, error=function(e) NULL)
ct <- sm$coeftable
rn <- rownames(ct)
sq_idx <- which(rn == 'I(meetings^2)')
sq_beta <- if (length(sq_idx) == 0) NA else unname(ct[sq_idx, 1])
sq_p <- if (length(sq_idx) == 0) NA else unname(ct[sq_idx, ncol(ct)])
# generic leads/lags
lag_idx <- grep('^lag[0-9]+_meetings$', rn)
lag_names <- rn[lag_idx]
lag_coefs <- if (length(lag_idx) == 0) list() else as.list(as.numeric(ct[lag_idx, 1]))
lag_pvals <- if (length(lag_idx) == 0) list() else as.list(as.numeric(ct[lag_idx, ncol(ct)]))
lead_idx <- grep('^lead[0-9]+_meetings$', rn)
lead_names <- rn[lead_idx]
lead_coefs <- if (length(lead_idx) == 0) list() else as.list(as.numeric(ct[lead_idx, 1]))
lead_pvals <- if (length(lead_idx) == 0) list() else as.list(as.numeric(ct[lead_idx, ncol(ct)]))
 # specific 1-step for backward-compat
l1_idx <- which(rn == 'lead1_meetings')
l1_beta <- if (length(l1_idx) == 0) NA else unname(ct[l1_idx, 1])
l1_p <- if (length(l1_idx) == 0) NA else unname(ct[l1_idx, ncol(ct)])
lg1_idx <- which(rn == 'lag1_meetings')
lg1_beta <- if (length(lg1_idx) == 0) NA else unname(ct[lg1_idx, 1])
lg1_p <- if (length(lg1_idx) == 0) NA else unname(ct[lg1_idx, ncol(ct)])
out <- list()
if (TRUE) {
  # base slope for ref domain (coefficient of 'meetings')
  base_idx <- which(rn == 'meetings')
  base_coef <- if (length(base_idx) == 0) NA else unname(ct[base_idx, 1])
  base_p <- if (length(base_idx) == 0) NA else unname(ct[base_idx, ncol(ct)])
  # deltas for other domains
  d_idx <- grepl(':meetings$', rn)
  d_rows <- rn[d_idx]
  d_names <- sub('^.*::', '', d_rows)
  d_names <- sub(':meetings$', '', d_names)
  d_coefs <- if (any(d_idx)) as.numeric(ct[d_idx, 1]) else numeric(0)
  d_pvals <- if (any(d_idx)) as.numeric(ct[d_idx, ncol(ct)]) else numeric(0)
  out$base_domain <- as.character(levels(as.factor(df$domain))[1])
  out$base_coef <- base_coef
  out$base_p <- base_p
  out$delta_domains <- as.list(d_names)
  out$delta_coefs <- as.list(d_coefs)
  out$delta_pvals <- as.list(d_pvals)
  out$n_obs <- as.integer(nobs(fit))
  out$collin_var <- as.list(cv)
  out$squared_coef <- sq_beta
  out$squared_p <- sq_p
  out$lead1_coef <- l1_beta
  out$lead1_p <- l1_p
  out$lag1_coef <- lg1_beta
  out$lag1_p <- lg1_p
  out$lag_names <- as.list(lag_names)
  out$lag_coefs <- lag_coefs
  out$lag_pvals <- lag_pvals
  out$lead_names <- as.list(lead_names)
  out$lead_coefs <- lead_coefs
  out$lead_pvals <- lead_pvals
} else {
  idx <- which(rn == 'meetings')
  beta <- if (length(idx) == 0) NA else unname(ct[idx, 1])
  p <- if (length(idx) == 0) NA else unname(ct[idx, ncol(ct)])
  out$beta <- beta
  out$p_value <- p
  out$n_obs <- as.integer(nobs(fit))
  out$squared_coef <- sq_beta
  out$squared_p <- sq_p
  out$lead1_coef <- l1_beta
  out$lead1_p <- l1_p
  out$lag1_coef <- lg1_beta
  out$lag1_p <- lg1_p
  out$lag_names <- as.list(lag_names)
  out$lag_coefs <- lag_coefs
  out$lag_pvals <- lag_pvals
  out$lead_names <- as.list(lead_names)
  out$lead_coefs <- lead_coefs
  out$lead_pvals <- lead_pvals
}
write(toJSON(out, auto_unbox=TRUE), file='C:/Users/caca_/AppData/Local/Temp/tmpbtod1qtp/ppml_output.json')
quit(status=0)
