data {
  int<lower=1> m;           // number of observations
  int<lower=1> k;           // number of coefficients
  real<lower=0, upper=1> t; // quantile of interes
  vector[m] y;              // response vector
  matrix[m,k] X;            // design matrix
  matrix[m,m] W;            // neighborhood matrix
}

transformed data {
  matrix[m,m] I = diag_matrix(rep_vector(1, m)); // generic identity matrix
}

parameters {
  real<lower=0> sigma_q;      // asymmetric laplace scale
  real<lower=-1,upper=1> phi; // spatial intensity
  vector[k] beta;             // vector of coefficientes
  vector<lower=0>[m] v;       // vector of latent exponential variables
}

transformed parameters {
 real a; // constant 'a' associated to the location-scale mixture
 real b; // constant 'b' associated to the location-scale mixture

 a = (1 - 2 * t) / (t * (1 - t)); // evaluation of 'a'
 b = 2 / (t * (1 - t));           // evaluation of 'b'
}

model {
 matrix[m,m] D = inverse(I - phi * W);                       // auxiliary matrix
 vector[m]   Mu = D * (X * beta + a * v);                    // response variable mean vector
 matrix[m,m] Sigma =  b * sigma_q * D * diag_matrix(v) * D'; // response variable covariance matrix
 
 v ~ exponential(pow(sigma_q, -1)); // location-scale mixture latent variable 'v'
 y ~ multi_normal(Mu, Sigma);       // response variable distribution
 
 beta ~ normal(0,100);         // coefficients prior distribution
 sigma_q ~ student_t(5,0,100); // location prior distribution
}

