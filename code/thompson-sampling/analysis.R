library(tidyverse)

dat = data.table::fread("thompson-scenarios.csv") %>%
  as_tibble() %>%
  mutate(lift_rel = factor(lift_rel, levels = c("0%", "10%", "100%"), ordered = T),
         theta_a_str = factor(
           theta_a_str,
           levels = paste("theta_a =", scales::percent(c(.01, .05, .10), .01)),
           ordered = TRUE))

dat %>%
  group_by(theta_a_str, theta_a, lift_rel, nth_user) %>%
  summarise_at(vars("b_is_better"), list(
    y = ~mean(.),
    ymax = ~quantile(., 0.9),
    ymin = ~quantile(., 0.1)
  )) %>%
  ungroup() %>%
  ggplot(aes(nth_user, y, ymin = ymin, ymax = ymax, fill = lift_rel)) +
  geom_line(lwd = 1, aes(color = lift_rel)) +
  geom_ribbon(alpha = .2) +
  geom_hline(yintercept = .5, lty = 2) +
  scale_x_continuous(name = "n = # of users") +
  scale_y_continuous(name = expression(paste(P(theta[B] > theta[A]), " = ", P(Delta > 0))),
                     labels = scales::percent,
                     breaks = seq(0, 1, by = .1),
                     limits = c(0,1)) +
  scale_color_manual(name = expression(Delta/theta[A]),
                     values = c("darkgrey", "blue4", "green4")) +
  scale_fill_manual(name = expression(Delta/theta[A]),
                    values = c("darkgrey", "blue4", "green4")) +
  facet_wrap(~ theta_a_str, scales = "fixed") +
  theme_bw(base_size = 15)

file_name = "~/abelborges.github.io/images/thompson-sampling/simple-scenarios.png"
ggsave(file_name, width = 12, height = 4)

# HDR
# source: https://sites.google.com/site/doingbayesiandataanalysis/software-installation
# discovered at: https://stackoverflow.com/a/42987104/6152355
get_hdr = function(icdf, cred_prob = 0.80, tol = 1e-8, ...) {
  interval_width = function(lower_tail_prob, icdf, cred_prob, ...)
    icdf(cred_prob + lower_tail_prob, ...) - icdf(lower_tail_prob, ...)
  
  hdr_lower_tail_prob = optimize(
    f = interval_width, interval = c(0 , 1 - cred_prob),
    icdf = icdf, cred_prob = cred_prob, tol = tol, ...)$minimum
  
  c(icdf(hdr_lower_tail_prob, ...), icdf(cred_prob + hdr_lower_tail_prob, ...))
}

# numerical descriptions of the distribution of Delta
delta_pdf = function(d, alpha_a, beta_a, alpha_b, beta_b) {
  integrate(function(theta) {
    dbeta(d + theta, alpha_b, beta_b) * dbeta(theta, alpha_a, beta_a)
  }, lower = 0, upper = 1)$value
}

delta_cdf = function(d, alpha_a, beta_a, alpha_b, beta_b) {
  integrate(Vectorize(delta_pdf), lower = -1, upper = d,
            alpha_a = alpha_a, beta_a = beta_a,
            alpha_b = alpha_b, beta_b = beta_b)$value
}

delta_icdf = function(p, alpha_a, beta_a, alpha_b, beta_b) {
  uniroot(function(d) delta_cdf(d, alpha_a, beta_a, alpha_b, beta_b) - p,
          interval = c(-1, 1))$root
}

# test
d = seq(-1, 1, by = .01)
pdf = sapply(d, delta_pdf, alpha_a=10, beta_a=20, alpha_b=10, beta_b=50)
plot(d, pdf,type='l')
hdr = get_hdr(delta_icdf, alpha_a=10, beta_a=20, alpha_b=10, beta_b=50)
abline(v = hdr, lty = 2)
