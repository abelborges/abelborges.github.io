library(dplyr)
library(tidyr)
library(readr)
Rcpp::sourceCpp("abelborges.github.io/code/thompson-sampling/simulations.cpp")
# setwd("/path/to/working/dir")

is_equal = function(x, y) abs(x - y) < 1e-10

simulations = function(n_users = 10000, replicates = 100, seed = 1,
                       theta_a, theta_b, file_name) {
  set.seed(seed)
  simulate_many(n_users, replicates, theta_a, theta_b) %>%
    lapply(as_tibble) %>%
    do.call(bind_rows, .) %>%
    write_csv(file_name)
}

# scenarios for base rate (theta_a) and lift (theta_b - theta_a)
simulations(theta_a = .10, theta_b = .10, file_name = "thompson-10-no-lift.csv")
simulations(theta_a = .10, theta_b = .11, file_name = "thompson-10-small-lift.csv")
simulations(theta_a = .10, theta_b = .20, file_name = "thompson-10-big-lift.csv")
simulations(theta_a = .010, theta_b = .010, file_name = "thompson-01-no-lift.csv")
simulations(theta_a = .010, theta_b = .011, file_name = "thompson-01-small-lift.csv")
simulations(theta_a = .010, theta_b = .020, file_name = "thompson-01-big-lift.csv")

# merge
expand.grid(x = c("10", "01"), y = c("no", "small", "big"), stringsAsFactors = F) %>%
  mutate(z = paste0("thompson-", x, "-", y, "-lift.csv")) %>%
  pull(z) %>%
  lapply(read_csv) %>%
  do.call(bind_rows, .) %>%
  mutate(
    lift = case_when(
      is_equal(lift/theta_a, 0.0) ~ "0%",
      is_equal(lift/theta_a, 0.1) ~ "10%",
      is_equal(lift/theta_a, 1.0) ~ "100%"
    ),
    theta_a = scales::percent(theta_a, .01)
  ) %>%
  write_csv("thompson-scenarios.csv")
