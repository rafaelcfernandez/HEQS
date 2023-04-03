data {
  int<lower=1> m; //total of counties
  int<lower=1> n; //total of schools
  int<lower=1> k; //total of counties covariates
  int<lower=1> r; //total of school covariates
  vector[n] y;    // school response vector
  matrix[n,m] H;  // point matrix
  matrix[n,r] G;  // school design matrix
  matrix[m,m] W;  // proximity matrix
  matrix[m,k] X;  // county design matrix
}

transformed data {
  matrix[m,m] Ic = diag_matrix(rep_vector(1, m)); // generic identity matrix
  matrix[n,n] Is = diag_matrix(rep_vector(1, n)); // generic identity matrix
}

parameters {
  vector[k] beta;             // school parameters
  vector[r] theta;            // county parameters
  vector[m] eta;              // latent city response
  real<lower=-1,upper=1> phi; // spatial parameter
  real<lower=0> sigma_c;      // county variability
  real<lower=0> sigma_s;      // school variability
}

model {
 eta ~ multi_normal(inverse(Ic - phi * W)*X*beta, sqrt(sigma_c)*inverse(Ic - phi * W)*inverse(Ic - phi * W)'); // city random effect distribution 
 y ~ multi_normal(H*eta + G*theta, sqrt(sigma_s)*Is);                                                          // response variable distribution
 
 beta ~ normal(0,100);         // city coefficients prior distribution
 theta ~ normal(0,100);        // school coefficients prior distribution
 phi ~ uniform(-1,1);          // spatial parameter prior distribution
 sigma_s ~ student_t(5,0,100); // school level scale parameter prior distribution
 sigma_c ~ student_t(5,0,100); // city level scale parameter prior distribution
}
