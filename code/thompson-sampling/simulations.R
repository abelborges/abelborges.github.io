library(dplyr)
library(readr)
Rcpp::sourceCpp("~/abelborges.github.io/code/thompson-sampling/simulations.cpp")
# setwd("/path/to/working/dir")
set.seed(1)

is_equal = function(x, y) abs(x - y) < 1e-10

simulations = function(theta_a, theta_b, file_name, n_users = 10000, replicates = 100) {
  simulate_many(n_users, replicates, theta_a, theta_b) %>%
    lapply(as_tibble) %>%
    do.call(bind_rows, .) %>%
    write_csv(file_name)
}

# scenarios
simulations(.10, .10, "thompson-10-no-lift.csv")
simulations(.10, .11, "thompson-10-small-lift.csv")
simulations(.10, .20, "thompson-10-big-lift.csv")

simulations(.050, .050, "thompson-05-no-lift.csv")
simulations(.050, .055, "thompson-05-small-lift.csv")
simulations(.050, .100, "thompson-05-big-lift.csv")

simulations(.010, .010, "thompson-01-no-lift.csv")
simulations(.010, .011, "thompson-01-small-lift.csv")
simulations(.010, .020, "thompson-01-big-lift.csv")

# merge scenarios
expand.grid(x = c("10", "05", "01"),
            y = c("no", "small", "big"),
            stringsAsFactors = F) %>%
  with(paste0("thompson-", x, "-", y, "-lift.csv")) %>%
  lapply(read_csv) %>%
  do.call(bind_rows, .) %>%
  mutate(
    lift_rel = case_when(
      is_equal((theta_b - theta_a)/theta_a, 0.0) ~ "0%",
      is_equal((theta_b - theta_a)/theta_a, 0.1) ~ "10%",
      is_equal((theta_b - theta_a)/theta_a, 1.0) ~ "100%"
    ),
    theta_a_str = paste("theta_a =", scales::percent(theta_a, .01))
  ) %>%
  write_csv("thompson-scenarios.csv")
