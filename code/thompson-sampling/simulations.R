# setwd("abelborges.github.io/code/thompson-sampling/")

library(dplyr)
library(readr)
library(foreach)
library(doParallel)
registerDoParallel(cores = 4)
Rcpp::sourceCpp("simulations.cpp")
set.seed(1)

is_equal = function(x, y) abs(x - y) < 1e-10

simulations = function(theta_a, theta_b, n_users = 10000, replicates = 100) {
  simulate_many(n_users, replicates, theta_a, theta_b) %>%
    lapply(as_tibble) %>%
    do.call(bind_rows, .)
}

# scenarios
scenarios = list(
  list(theta_a = .10, theta_b = .10),
  list(theta_a = .10, theta_b = .11),
  list(theta_a = .10, theta_b = .20),
  
  list(theta_a = .050, theta_b = .050),
  list(theta_a = .050, theta_b = .055),
  list(theta_a = .050, theta_b = .100),
  
  list(theta_a = .010, theta_b = .010),
  list(theta_a = .010, theta_b = .011),
  list(theta_a = .010, theta_b = .020)
)

results = foreach(scenario = scenarios) %dopar% {
  simulations(scenario$theta_a, scenario$theta_b, scenario$file_name)
}

# write combined results
results %>%
  do.call(bind_rows, .) %>%
  mutate(
    r = theta_b/theta_a - 1,
    lift_rel = case_when(
      is_equal(r, 0.0) ~ "0%",
      is_equal(r, 0.1) ~ "10%",
      is_equal(r, 1.0) ~ "100%"
    ),
    theta_a_str = paste("theta_a =", scales::percent(theta_a, .01))
  ) %>%
  select(-r) %>%
  write_csv("thompson-scenarios.csv")
