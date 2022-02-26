data {
  int<lower=1> n; //total of counties
  int<lower=1> m; //total of schools
  int<lower=1> k; //total of counties covariates
  int<lower=1> r; //total of school covariates
  vector[m] ys; // school response vector
  matrix[m,n] H; // point matrix
  matrix[m,r] G; // school design matrix
  matrix[n,n] W; // proximity matrix
  matrix[n,k] X; // county design matrix
}

transformed data {
  matrix[n,n] Ic = diag_matrix(rep_vector(1, n)); // generic identity matrix
  matrix[m,m] Is = diag_matrix(rep_vector(1, m)); // generic identity matrix
}

parameters {
  vector[k] beta; // school parameters
  vector[r] theta; // county parameters
  vector[n] yc; // latent school response
  real<lower=-1,upper=1> phi; // spatial parameter
  real<lower=0> sigma_c; // county variability
  real<lower=0> sigma_s; // school variability
}

model {
 yc ~ multi_normal(inverse(Ic - phi * W)*X*beta, sqrt(sigma_c)*inverse(Ic - phi * W)*inverse(Ic - phi * W)'); 
 ys ~ multi_normal(H*yc + G*theta, sqrt(sigma_s)*Is);
 beta ~ normal(0,100);
 theta ~ normal(0,100);
 phi ~ uniform(-1,1);
 sigma_s ~ student_t(5,0,100);
 sigma_c ~ student_t(5,0,100);
}
