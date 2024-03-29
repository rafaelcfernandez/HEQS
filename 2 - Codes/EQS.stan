functions { // function associated to the generalized laplace asymmetric distribution
  real g(real gamma) {
    real gg; 
    gg = 2 * normal_cdf(-fabs(gamma),0,1) * exp(gamma^2 / 2);
    return gg;
  }
}

data {
  int<lower=1> m;            // number of observations
  int<lower=1> k;            // number of coefficients
  real L;                    // g-, gamma support lower bound
  real U;                    // g+, gamma support upper bound
  real<lower=0, upper=1> t0; // quantile of interest
  vector[m] y;               // response variable
  matrix[m,k] X;             // design matrix
  matrix[m,m] W;             // neighborhood matrix
}

transformed data {
  matrix[m,m] I = diag_matrix(rep_vector(1, m)); // generic identity matrix
}

parameters {
  vector[k] beta;             // vector of coefficients
  real<lower=-1,upper=1> phi; // spatial insentisy parameter
  real<lower=0> sigma;        // generalized assymetric laplace scale parameter
  real gamma;                 // generalized assymetric laplace shape parameter
  vector<lower=0>[m] v;       // vector of latent exponential variables
  vector<lower=0>[m] z;       // vector of latent truncated normal variables
}

transformed parameters {
 real<lower=0,upper=1> t; // reparametrized quantile order
 real a;                  // constant 'a' associated to the location-scale mixture
 real<lower=0> b;         // constant 'b' associated to the location-scale mixture
 real c;                  // constant 'c' associated to the location-scale mixture

 t =  (gamma<0) + (t0-(gamma<0))/g(gamma); // evaluation of 't'
 a = (1 - 2 * t) / (t * (1 - t));          // evaluation of 'a'
 b = 2 / (t * (1 - t));                    // evaluation of 'b'
 c = pow((gamma>0)-t,-1);                  // evaluation of 'c'
}

model {
 matrix[m,m] D = inverse(I - phi * W);                        // auxiliary matrix
 vector[m]   Mu = D * (X*beta + a*v + sigma*c*fabs(gamma)*z); // response variable mean vector
 matrix[m,m] Sigma =  b * sigma * D * diag_matrix(v) * D';    // response variable covariance matrix
  
 z ~ normal(0,1);                 // location-scale mixture latent variable 'z'
 v ~ exponential(pow(sigma, -1)); // location-scale mixture latent variable 'v'
 
 y ~ multi_normal(Mu, Sigma);     // response variable distribution
 
 gamma ~ uniform(L,U);            // shape parameter prior distribution
 beta ~ normal(0,100);            // coefficients prior distribution
 phi ~ uniform(-1,1);             // spatial parameter prior distribution
 sigma ~ cauchy(0,100);           // scale parameter prior distribution
}
