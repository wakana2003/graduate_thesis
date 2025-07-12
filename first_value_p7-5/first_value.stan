data {
  int<lower=1> N;
  int<lower=1> T;
  array[N, T] int<lower=0, upper=1> choice;
  array[N, T] real reward_A;
  array[N, T] real reward_B;
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0>          beta;
  real                   Q_init1;
}

transformed parameters {
  real Q_init2 = 0;
}

model {
  alpha   ~ beta(2, 2);
  beta    ~ lognormal(log(1), 0.5);
  Q_init1 ~ normal(3.75, 2);

  for (n in 1:N) {
    vector[2] Q = rep_vector(0.0, 2);
    Q[1] = Q_init1;
    Q[2] = Q_init2;
    for (t in 1:T) {
      real pA = inv_logit(beta * (Q[1] - Q[2]));
      target += bernoulli_lpmf(1 - choice[n, t] | pA);
      int a = choice[n, t] + 1;
      real r = (a == 1) ? reward_A[n, t] : reward_B[n, t];
      Q[a] += alpha * (r - Q[a]);
    }
  }
}

generated quantities {
  array[N, T] int<lower=0,upper=1> choice_rep;
  array[N, T] real              pA_track;

  for (n in 1:N) {
    vector[2] Q = rep_vector(0.0, 2);
    Q[1] = Q_init1;
    Q[2] = Q_init2;
    for (t in 1:T) {
      real pA = inv_logit(beta * (Q[1] - Q[2]));
      pA_track[n, t]   = pA;
      choice_rep[n, t] = 1 - bernoulli_rng(pA);
      int a = choice_rep[n, t] + 1;
      real r = (a == 1) ? reward_A[n, t] : reward_B[n, t];
      Q[a] += alpha * (r - Q[a]);
    }
  }
}
