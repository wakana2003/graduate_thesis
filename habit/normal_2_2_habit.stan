data {
  int<lower=1> N;
  int<lower=1> T;
  array[N, T] int<lower=0, upper=1> choice;
  array[N, T] real reward_A;
  array[N, T] real reward_B;
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0> beta;
}

model {
  alpha ~ beta(2, 2);
  beta  ~ lognormal(log(1), 0.5);

  for (n in 1:N) {
    vector[2] Q = [0, 0]';

    for (t in 1:T) {
      real pA = inv_logit(beta * (Q[1] - Q[2]));
      int is_A = 1 - choice[n, t];
      is_A ~ bernoulli(pA);

      int a = choice[n, t] + 1;
      real r = (a == 1) ? reward_A[n, t] : reward_B[n, t];
      Q[a] += alpha * (r - Q[a]);
    }
  }
}

generated quantities {
  array[N, T] int choice_rep;
  array[N, T] real pA_track;

  for (n in 1:N) {
    vector[2] Q = [0, 0]';
    for (t in 1:T) {
      real pA = inv_logit(beta * (Q[1] - Q[2]));
      pA_track[n, t] = pA;
      choice_rep[n, t] = 1 - bernoulli_rng(pA);

      int a = choice_rep[n, t] + 1;
      real r = (a == 1) ? reward_A[n, t] : reward_B[n, t];
      Q[a] += alpha * (r - Q[a]);
    }
  }
}
