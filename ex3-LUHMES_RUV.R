library(sva)
library(ruv)


#' Parse the formula in \code{cate}, and return a matrix of primary variables and a matrix of nuisance variables.
#' @keywords internal
parse.cate.formula <- function(formula, X.data = NULL) {

    formula <- as.formula(formula)
    parse.fm <- as.character(formula)

    if (length(parse.fm) == 2) {
    	covariates <- parse.fm[2] # the formula looks like "~x"
    } else if (length(parse.fm) == 3) {
    	covariates <- parse.fm[3] # the formula looks like "y ~ x"
    }	else {
    	stop("Unknown formula pattern!")
    }

    covariates <- unlist(strsplit(covariates, "[|]"))
    if (length(covariates) > 2) {
    	stop("The formula should not contain more than one |!")
    }
    if (length(covariates) == 1) {
    	X.primary <- model.matrix(as.formula(paste("~", covariates)), data = X.data)
        X.nuis <- NULL
    } else {
        ## If there is |, always put interecept in X.nuis
    	X.primary <- model.matrix(as.formula(paste("~", covariates[1])), data = X.data)[, -1, drop = FALSE]
    	X.nuis <- model.matrix(as.formula(paste("~", covariates[2])), data = X.data)
    }

    return(list(X.primary = X.primary, X.nuis = X.nuis))
}

#' Check linear dependence
#'
#' Checks if \code{X.primary} has full rank and is linearly independent of \code{x.nuis}.
#'
#' @details X.nuis is allowed to not have full rank.
#' @keywords internal
#'
check.rank <- function(X.primary, X.nuis) {
    if (Matrix::rankMatrix(X.primary) < ncol(X.primary)) {
        stop("X.primary is not full rank!")
    }
    r0 <- 0
    if (!is.null(X.nuis)) {
        r0 <- Matrix::rankMatrix(X.nuis)
    }
    if (Matrix::rankMatrix(cbind(X.nuis, X.primary)) != ncol(X.primary) + r0) {
        stop("Some columns in X.primary linearly depend on X.nuis!")
    }
}


#' @rdname wrapper
#' @param sva.method parameter for \code{\link[sva]{sva}}.
#'        whether to use an iterative reweighted algorithm (irw) or a two-step algorithm (two-step).
#' @param B parameter for \code{\link[sva]{sva}}. the number of iterations of the irwsva algorithm
#' @details The \code{beta.p.values} returned is a length \code{p} vector, each for the overall effects of all the primary variables.
#'
#' @import sva stats
#' @export
#'
sva.wrapper <- function(formula,
                        X.data = NULL,
                        Y,
                        r,
                        sva.method = c("irw", "two-step"),
                        B = 5) {

    method <- match.arg(sva.method, c("irw", "two-step"))
    dat <- t(Y)

    X <- parse.cate.formula(formula, X.data)
    X.primary <- X$X.primary
    X.nuis <- X$X.nuis

    check.rank(X.primary, X.nuis)

    mod <- cbind(X.primary, X.nuis)
    mod0 <- X.nuis
    if (ncol(X.nuis) == 0)
    	mod0 <- NULL

    result <- sva(dat, mod, mod0, r, method = method, B= B)
    Z <- result$sv
    rownames(Z) <- rownames(X)
    modSV <- cbind(mod, Z)
    mod0SV <- cbind(mod0, Z)
    ## only can calculate a vector of p-values. It's the p-values of the effect of all the primary variables as a whole
    p.values <- f.pvalue(dat, modSV, mod0SV)
    return(list(beta.p.value = p.values, Z= Z, beta.p.post = result$pprob.b))

}


#' @rdname wrapper
#' @param ruv.method either using \code{\link[ruv:RUV2]{RUV2}}, \code{\link[ruv:RUV4]{RUV4}} or
#'    	         \code{\link[ruv:RUVinv]{RUVinv}} functions
#' @param nc parameter for \link{ruv} functions: position of the negative controls
#' @param lambda parameter for \code{\link[ruv:RUVinv]{RUVinv}}
#'
#' @import ruv
#' @export
#'
ruv.wrapper <- function(
                        X.primary,
                        X.nuis,
                        Y,
                        r,
                        nc,
                        lambda = 1,
                        ruv.method = c("RUV2", "RUV4", "RUVinv"), ...) {

    method <- match.arg(ruv.method, c("RUV2", "RUV4", "RUVinv"))

    # X <- parse.cate.formula(formula, X.data)
    # X.primary <- X$X.primary
    # X.nuis <- X$X.nuis

    check.rank(X.primary, X.nuis)

    X <- X.primary
    Z <- X.nuis
    if (ncol(X.nuis) == 0)
    	Z <- NULL

    p <- ncol(Y)
    n <- nrow(Y)
    ctl <- rep(F, p)
    ctl[nc] <- T
    if (method == "RUV2"){
        result <- RUV2(Y, X, ctl, r, Z, ...)
    } else if (method == "RUV4") {
        result <- RUV4(Y, X, ctl, r, Z, ...)
    } else if (method == "RUVinv") {
        ## uses p - r0 - r1 latent factors!
        if (length(nc) > n) {
            result <- RUVinv(Y, X, ctl, Z, ...)
        } else
            result <- RUVinv(Y, X, ctl, Z, lambda = lambda, ...)
    }
    beta <- t(result$betahat)
    Gamma <- t(result$alpha/sqrt(n))
    beta.t <- t(result$t)
    beta.p.value <- t(result$p)
    Sigma <- result$sigma2
    names(Sigma) <- colnames(Y)
    return(list(Gamma = Gamma, Sigma = Sigma, beta = beta, beta.t = beta.t,
                beta.p.value = beta.p.value, Z = result$W))
}






library(hdf5r)

# Read datasets from the HDF5 file
file.h5 <- H5File$new("data/LUHMES/data.h5", mode = "r")
gene_names <- file.h5[["gene_names"]][]
Y <- t(file.h5[["Y"]][,])
colnames(Y) <- gene_names
cov_names <- file.h5[["cov_names"]][]
X <- data.frame(t(file.h5[["X"]][,]))
colnames(X) <- cov_names
var_features <- data.frame(t(file.h5[["var_features"]][,]))
colnames(var_features) <- c('vst.mean', 'vst.variance', 'vst.variance.expected', 'vst.variance.standardized', 'vst.variable')
file.h5$close_all()

path_rs <- "results/LUHMES/"
cov_test <- c('pt_state', cov_names[-c(1:6)])
cov_nui <- cov_names[! (cov_names %in% cov_test)]
nc <- var_features$vst.variance.standardized < 1
nc <- order(var_features$vst.variance.standardized)[1:4000]

setwd('cate/R/')
source('cate.R')
source('adjust_functions.R')
source('factor_functions.R')
setwd('../../')
library(MASS)
library(esaBcv)
library(ggplot2)


O <- t(qr.Q(qr(cbind(X[cov_nui], X[cov_test])), complete = TRUE))
Y.tilde <- O %*% Y


p <- est.factor.num(Y.tilde[-(1:length(cov_names)), ],
                method = "bcv", rmax = 80, nRepeat = 20, bcv.plot = TRUE)

# Create a data frame
res <- data.frame(
  r = 1:length(p$errors),
  errors = p$errors
)

# Create the plot using ggplot2
p <- ggplot(res, aes(x = r, y = errors)) +
  geom_point() +
  geom_line() +
  scale_y_log10() +  # Apply log scale to y-axis if needed
  labs(x = "r", y = "bcv MSE relative to the noise") +
  theme_minimal()
ggsave(sprintf("%snum_factors.pdf", path_rs), plot = p, width = 6, height = 6, dpi = 300)


for(r in c(5,10,20,50)){
    output <- ruv.wrapper(
        X[cov_test], X[cov_nui], Y, 
        r = r, nc = nc, ruv.method = 'RUV4', include.intercept = F)
    saveRDS(output, file = sprintf("%soutput_RUV4_%d.rds", path_rs, r))

    file.h5 <- H5File$new(sprintf("%soutput_RUV4_%d.h5", path_rs, r), mode = "w")
    file.h5[["beta"]] <- output$beta
    file.h5[["t_values"]] <- output$beta.t
    file.h5[["p_values"]] <- output$beta.p.value
    file.h5[["U_hat"]] <- output$Z
    file.h5$close_all()
}




for(r in c(5,10,20,50)){
    output <- cate.fit(
        X[cov_test], X[cov_nui], Y, 
        r = r, nc = nc, adj.method ='nc', calibrate=F)
    saveRDS(output, file = sprintf("%soutput_CATE_%d.rds", path_rs, r))

    file.h5 <- H5File$new(sprintf("%soutput_CATE_%d.h5", path_rs, r), mode = "w")
    file.h5[["beta"]] <- output$beta
    file.h5[["t_values"]] <- output$beta.t
    file.h5[["p_values"]] <- output$beta.p.value
    file.h5[["U_hat"]] <- output$Z
    file.h5$close_all()
}



for(r in c(5,10,20,50)){
    output <- cate.fit(
        X[cov_test], X[cov_nui], Y, 
        r = r, calibrate=F)
    saveRDS(output, file = sprintf("%soutput_CATEs_%d.rds", path_rs, r))

    file.h5 <- H5File$new(sprintf("%soutput_CATEs_%d.h5", path_rs, r), mode = "w")
    file.h5[["beta"]] <- output$beta
    file.h5[["t_values"]] <- output$beta.t
    file.h5[["p_values"]] <- output$beta.p.value
    file.h5[["U_hat"]] <- output$Z
    file.h5$close_all()
}
