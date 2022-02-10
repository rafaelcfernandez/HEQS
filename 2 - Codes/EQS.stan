functions {
  real g(real gamma) {
    real gg; 
    gg = 2 * normal_cdf(-fabs(gamma),0,1) * exp(gamma^2 / 2);
    return gg;
  }
}

data {
  int<lower=1> N; //number of observations
  int<lower=1> K; //number of coefficients
  real L; //g-, gamma support lower bound
  real U; //g+, gamma support upper bound
  real<lower=0, upper=1> t0; //quantile of interes
  vector[N] y; //response variable
  matrix[N,K] X; //design matrix
  matrix[N,N] W; //neighborhood matrix
}

transformed data {
  matrix[N,N] I = diag_matrix(rep_vector(1, N)); //generic identity matrix
}

parameters {
  vector[K] B; //vector of coefficients
  real<lower=-1,upper=1> phi; //spatial insentisy parameter
  real<lower=0> sigma; //generalized assymetric laplace scale parameter
  real gamma; //generalized assymetric laplace shape parameter
  vector<lower=0>[N] s; //vector of latent exponential variables
  vector<lower=0>[N] z; //vector of latent truncated normal variables
}

transformed parameters {
 real<lower=0,upper=1> t;
 real a; 
 real<lower=0> b;
 real c;

 t =  (gamma<0) + (t0-(gamma<0))/g(gamma);
 a = (1 - 2 * t) / (t * (1 - t));
 b = 2 / (t * (1 - t));
 c = pow((gamma>0)-t,-1);
}

model {
 matrix[N,N] D = inverse(I - phi * W);
 vector[N]   Mu = D * (X*B + a*s + sigma*c*fabs(gamma)*z);
 matrix[N,N] Sigma =  b * sigma * D * diag_matrix(s) * D';
  
 for (n in 1:N){ z[n] ~ normal(0,1)T[0,]; }
 gamma ~ uniform(L,U);
 s ~ exponential(pow(sigma, -1));
 y ~ multi_normal(Mu, Sigma);
 B ~ normal(0,100);
 phi ~ uniform(-1,1);
 sigma ~ cauchy(0,100);
}
