library(dplyr)
library(readr)
library(ggplot2)
# setwd("/path/to/working/dir")

dat = read_csv("thompson-scenarios.csv") %>%
  mutate(lift = factor(lift, levels = c("0%", "10%", "100%"), ordered = T))

dat %>%
  group_by(theta_a, lift, nth_user) %>%
  summarise_at(vars("b_is_better"), list(
    y = ~mean(.),
    ymax = ~quantile(., 0.9),
    ymin = ~quantile(., 0.1)
  )) %>%
  ungroup() %>%
  mutate(theta_a = paste0("theta_A = ", theta_a)) %>%
  ggplot(aes(nth_user, y, ymin = ymin, ymax = ymax, fill = lift)) +
  geom_line(lwd = 1, aes(color = lift)) +
  geom_ribbon(alpha = .2) +
  geom_hline(yintercept = .5, lty = 2) +
  scale_x_continuous(name = "# of users") +
  scale_y_continuous(name = expression(P(theta[B] > theta[A])),
                     labels = scales::percent,
                     breaks = seq(0, 1, by = .1),
                     limits = c(0,1)) +
  scale_color_manual(name = "Rel. lift", values = c("red4", "yellow4", "green4")) +
  scale_fill_manual(name = "Rel. lift", values = c("red4", "yellow4", "green4")) +
  facet_wrap(~ theta_a) +
  theme_bw(base_size = 15)

file_name = "~/abelborges.github.io/images/thompson-sampling/simple-scenarios.png"
ggsave(file_name, width = 10, height = 4)
