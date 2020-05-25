#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double prob_b_better_than_a(int alpha_a, int beta_a, int alpha_b, int beta_b) {
  double p = 0.0;
  for(int i = 0; i < alpha_b; i++) {
    p += exp(R::lbeta(alpha_a + i, beta_a + beta_b) -
      log(beta_b + i) - R::lbeta(1 + i, beta_b) - R::lbeta(alpha_a, beta_a));
  }
  return p;
}

// [[Rcpp::export]]
List simulate(
    int users, double theta_a, double theta_b, int universe,
    int alpha_a = 1, int beta_a = 1, int alpha_b = 1, int beta_b = 1) {
  NumericVector p_B(users), aa(users), ba(users), ab(users), bb(users);
  aa[0] = alpha_a;
  ba[0] = beta_a;
  ab[0] = alpha_b;
  bb[0] = beta_b;
  bool is_positive_outcome = true;
  NumericVector u1 = runif(users), u2 = runif(users);
  
  for(int i = 0; i < users; i++) {
    p_B[i] = prob_b_better_than_a(alpha_a, beta_a, alpha_b, beta_b);
    if(u1[i] < p_B[i]) { // B
      is_positive_outcome = u2[i] < theta_b;
      alpha_b += (is_positive_outcome ? 1 : 0);
      beta_b += (is_positive_outcome ? 0 : 1);
    } else { // A
      is_positive_outcome = u2[i] < theta_a;
      alpha_a += (is_positive_outcome ? 1 : 0);
      beta_a += (is_positive_outcome ? 0 : 1);
    }
    aa[i] = alpha_a;
    ba[i] = beta_a;
    ab[i] = alpha_b;
    bb[i] = beta_b;
  }
  
  return List::create(
    Named("theta_a") = theta_a,
    Named("theta_b") = theta_b,
    Named("universe") = universe,
    Named("nth_user") = seq(1, users),
    Named("b_is_better") = p_B,
    Named("alpha_a") = aa,
    Named("beta_a") = ba,
    Named("alpha_b") = ab,
    Named("beta_b") = bb
  );
}

// [[Rcpp::export]]
List simulate_many(int users, int reps, double theta_a, double theta_b) {
  List many(reps);
  for(int r = 0; r < reps; r++) {
    if(r % 10 == 0) std::cout << ((double) r)/reps << std::endl;
    many[r] = simulate(users, theta_a, theta_b, r+1);
  }
  return many;
}
