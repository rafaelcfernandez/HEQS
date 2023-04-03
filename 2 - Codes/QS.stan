data {
  int<lower=1> j;           //number of observations
  int<lower=1> k;           //number of coefficients
  real<lower=0, upper=1> t; //quantile of interes
  vector[j] y;              //response vector
  matrix[j,k] X;            //design matrix
  matrix[j,j] W;            //neighborhood matrix
}

transformed data {
  matrix[j,j] I = diag_matrix(rep_vector(1, j)); //generic identity matrix
}

parameters {
  real<lower=0> sigma_q;      //asymmetric laplace scale
  real<lower=-1,upper=1> phi; //spatial intensity
  vector[k] beta;             //vector of coefficientes
  vector<lower=0>[j] v;       //vector of latent exponential variables
}

transformed parameters {
 real a; //constant 'a' associated to the location-scale mixture
 real b; //constant 'b' associated to the location-scale mixture

 a = (1 - 2 * t) / (t * (1 - t));
 b = 2 / (t * (1 - t));
}

model {
 matrix[j,j] D = inverse(I - phi * W);
 vector[j]   Mu = D * (X * beta + a * v);
 matrix[j,j] Sigma =  b * sigma_q * D * diag_matrix(v) * D';
 
 v ~ exponential(pow(sigma_q, -1));
 y ~ multi_normal(Mu, Sigma);
 
 beta ~ normal(0,100);
 sigma_q ~ student_t(5,0,100);
}

