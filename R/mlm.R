#' Bayesian parameter estimation for multilevel mediation models
#'
#' Easy Bayesian parameter estimation for multilevel mediation models.
#'
#' @param d A \code{data.frame}.
#' @param id Column of participant IDs in \code{data}.
#' @param x Column of X values in \code{data}.
#' @param m Column of M values in \code{data}.
#' @param y Column of Y values in \code{data}.
#' @param ... Other optional parameters passed to \code{rstan::stan()}.
#'
#' @return An object of S4 class stanfit, with all its available methods.
#'
#' @author Matti Vuorre \email{mv2521@columbia.edu}
#'
#' @details Draw samples from the joint posterior distribution of a
#' multilevel mediation model using Stan.
#'
#' By default, the procedure doesn't return all the estimated parameters,
#' such as covariances etc. To return all estimated parameters,
#' set \code{everything = TRUE}.
#'
#' @examples
#' \dontrun{
#' ## Run example from Bolger and Laurenceau (2013)
#' data(BLch9)
#' fit <- mlm(BLch9)
#' print(fit)}
#'
#' @import rstan
#' @export

mlm <- function(d = NULL, id = id, x = x, m = m, y = y, ...) {

    # Check for data and quit if not suitable
    if (!(is.data.frame(d))) stop("d is not a data.frame")

    # Create a data list for Stan
    standat <- with(d, list(
        N = nrow(d),
        X = x,
        M = m,
        Y = y,
        id = seq_ids(id),  # Coerces ids to 1:J
        J = length(unique(id))
    ))

    # Sample from model
    model_file <- system.file("stan/bmlm.stan", package="bmlm")
    message("Estimating model, please wait.")
    fit <- rstan::stan(file = model_file,
                       model_name = "Multilevel mediation",
                       data = standat, ...)

    # Return stanfit object
    return(fit)
}