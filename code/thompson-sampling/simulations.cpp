#include <Rcpp.h>
using namespace Rcpp;

double B(size_t a, size_t b) {
  return exp(R::lbeta((double) a, (double) b));
}

// [[Rcpp::export]]
double prob_b_better_than_a(size_t alpha_a, size_t beta_a, size_t alpha_b, size_t beta_b) {
  double p = 0.0;
  for(size_t i = 0; i < alpha_b; i++) {
    p += B(alpha_a + i, beta_a + beta_b) /
      ((beta_b + i) * B(1 + i, beta_b) * B(alpha_a, beta_a));
  }
  return p;
}

double u() { return runif(1)[0]; }

// [[Rcpp::export]]
List simulate(size_t users) {
  size_t alpha_a = 1, beta_a = 1, alpha_b = 1, beta_b = 1;
  NumericVector p_B(users);

  for(size_t i = 0; i < users; i++) {
    p_B[i] = prob_b_better_than_a(alpha_a, beta_a, alpha_b, beta_b);
    if(u() < p_B[i]) { // B
      
    } else { // A

    }
  }

  return List::create(
    Named("p") = p_B
  );
}
