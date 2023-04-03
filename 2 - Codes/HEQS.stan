functions { // function associated to the generalized laplace asymmetric distribution
  real g(real gamma) {
    real gg; 
    gg = 2 * normal_cdf(-fabs(gamma),0,1) * exp(gamma^2 / 2); 
    return gg;
  }
}

data {
  int<lower=1> m;            // total of cities
  int<lower=1> n;            // total of schools
  int<lower=1> k;            // total of cities covariates
  int<lower=1> r;            // total of school covariates
  real L;                    // gamma lower bound
  real U;                    // gamma upper bound
  real<lower=0, upper=1> t0; // quantile of interest
  vector[n] y;               // school response vector
  matrix[n,m] H;             // point matrix
  matrix[n,r] G;             // school design matrix
  matrix[m,m] W;             // proximity matrix
  matrix[m,k] X;             // city design matrix
}

transformed data {
  matrix[m,m] I = diag_matrix(rep_vector(1, m)); // generic identity matrix
}

parameters {
  vector[k] beta;              // city parameters
  vector[r] theta;             // school parameters
  vector[m] eta;               // latent city effect
  real<lower=-1,upper=1> phi;  // spatial parameter
  real<lower=L,upper=U> gamma; // shape parameter
  real<lower=0> sigma2_s;      // city variability
  real<lower=0> sigma_q;       // school variability
  vector<lower=0>[n] v;        // latent variable
  vector<lower=0>[n] z;        // latent variable
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
 z ~ normal(0,1);                   // location-scale mixture latent variable 'z'
 v ~ exponential(pow(sigma_q, -1)); // location-scale mixture latent variable 'v'
 
 eta ~ multi_normal(inverse(I - phi * W)*X*beta,sigma2_s*inverse(I - phi * W)*inverse(I - phi * W)'); // city random effect distribution
 y ~ multi_normal(H*eta + G*theta + a*v + sigma_q*c*fabs(gamma)*z, b*sigma_q*diag_matrix(v));         // response variable distribution

 beta ~ normal(0,100);          // city coefficients prior distribution
 theta ~ normal(0,100);         // school coefficients prior distribution
 phi ~ uniform(-1,1);           // spatial parameter prior distribution
 sigma_q ~ student_t(5,0,100);  // school level scale parameter prior distribution
 sigma2_s ~ student_t(5,0,100); // city level scale parameter prior distribution
 gamma ~ student_t(1,0,1);      // shape parameter prior distribution
}
