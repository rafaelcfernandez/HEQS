functions {
  real g(real gamma) {
    real gg; 
    gg = 2 * normal_cdf(-fabs(gamma),0,1) * exp(gamma^2 / 2); 
    return gg;
  }
}

data {
  int<lower=1> n; //total of cities
  int<lower=1> m; //total of schools
  int<lower=1> k; //total of cities covariates
  int<lower=1> r; //total of school covariates
  real L; // gamma lower bound
  real U; // gamma upper bound
  real<lower=0, upper=1> t0; //quantile of interest
  vector[m] yl; // school response vector
  matrix[m,n] H; // point matrix
  matrix[m,r] G; // school design matrix
  matrix[n,n] W; // proximity matrix
  matrix[n,k] X; // city design matrix
}

transformed data {
  matrix[n,n] I = diag_matrix(rep_vector(1, n)); // generic identity matrix
}

parameters {
  vector[k] beta; // school parameters
  vector[r] theta; // city parameters
  vector[n] yu; // latent school response
  real<lower=-1,upper=1> phi; // spatial parameter
  real<lower=L,upper=U> gamma; // shape parameter
  real<lower=0> sigma2_s; // city variability
  real<lower=0> sigma_q; // school variability
  vector<lower=0>[m] v; // latent variable
  vector<lower=0>[m] z; // latent variable
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
 z ~ normal(0,1); 
 v ~ exponential(pow(sigma_q, -1));
 yu ~ multi_normal(inverse(I - phi * W)*X*beta,sigma2_s*inverse(I - phi * W)*inverse(I - phi * W)'); 
 yl ~ multi_normal(H*yu + G*theta + a*v + sigma_q*c*fabs(gamma)*z, b*sigma_q*diag_matrix(v));
 beta ~ normal(0,100);
 theta ~ normal(0,100);
 phi ~ uniform(-1,1);
 sigma_q ~ student_t(5,0,100);
 sigma2_s ~ student_t(5,0,100);
 gamma ~ student_t(1,0,1);
}
