data {
  int<lower=1> n; //number of observations
  int<lower=1> k; //number of coefficients
  real<lower=0, upper=1> t; //quantile of interes
  vector[n] y; //response vector
  matrix[n,k] X; //design matrix
  matrix[n,n] W; //neighborhood matrix
}

transformed data {
  matrix[n,n] I = diag_matrix(rep_vector(1, n)); //generic identity matrix
}

parameters {
  real<lower=0> sigma_q; //asymmetric laplace scale
  real<lower=-1,upper=1> phi; //spatial intensity
  vector[k] beta; //vector of coefficientes
  vector<lower=0>[n] v; //vector of latent exponential variables
}

transformed parameters {
 real a; 
 real b; 

 a = (1 - 2 * t) / (t * (1 - t));
 b = 2 / (t * (1 - t));
}

model {
 matrix[n,n] D = inverse(I - phi * W);
 vector[n]   Mu = D * (X * beta + a * v);
 matrix[n,n] Sigma =  b * sigma_q * D * diag_matrix(v) * D';
 
 v ~ exponential(pow(sigma_q, -1));
 y ~ multi_normal(Mu, Sigma);
 
 beta ~ normal(0,100);
 sigma_q ~ student_t(5,0,100);
}
 
