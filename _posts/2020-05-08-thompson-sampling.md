---
layout: post
title: "A fairly comprehensive introduction to online experiments with Thompson Sampling"
excerpt: >
  We're going to release a new feature in a product.
  It consists of a modified version B of a currently functional component A.
  Version B is technically OK but we want to check whether users prefer it over A
  and, if so, gradually point new requests to it.
  This post is about a solution to this problem.
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

We're going to release a new feature in a web application.
It consists of a modified version B of a currently functional component A.
Version B is technically OK, just as A,
but we want to discover which one users prefer
and direct requests accordingly.
This post is about a solution to this problem.

One way to check that is by carrying an A/B test.
For a frequentist-type methodology for testing, you may
check [this]({{site.baseurl}}/hypothesis-testing).
Here I want to talk about a Bayesian approach to update
our impressions on A and B in an online fashion
known as Thompson Sampling.

## Bayesian inference

Let's say we're interested in a random process $$X$$
whose realizations are supposed to follow a probability
distribution $$p(x \mid \theta)$$,
parametrized by a vector $$\theta$$
that takes values from a set $$\Theta$$ (of, say, real vectors).
The function $$\theta \mapsto p(x \mid \theta)$$
is called the **likelihood** of $$\theta$$;
it's just another name for the density/probability mass of $$x$$
understood as a function of $$\theta$$ and evaluated
at $$x$$ assuming that the data was already observed.

Then, inferences about the behavior of $$X$$
(issues like which values are most likely to occur,
how frequently the values lie on a given interval, and so on)
can be translated into inferences about $$\theta$$,
which we assume to be a constant that we don't know for sure.

Simply put, the Bayesian approach for inference is

1. To model our prior perceptions and uncertainties
of what the value of $$\theta$$ may be with a probability
distribution, say $$p(\theta)$$, known as the
**prior** distribution;
1. And, then, once we observe data $$x$$ from $$X$$, use the
[Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
to deduce the **posterior**
distribution of $$\theta$$ given this piece of evidence,
namely

$$
p(\theta \mid x)
= \frac{p(x \mid \theta) p(\theta)}
{\int_\Theta p(x \mid \theta) p(\theta) d \theta}
\propto p(x \mid \theta) p(\theta).
$$

For most interesting problems, the denominator may be very difficult
to compute exactly in closed form, or even via numerical methods.
It's just the normalizing constant that guarantees
that $$p(\theta \mid x)$$ integrates to 1.
The harshness of obtaining a clear picture of the posterior
in a given setup (likelihood + prior + data) is a problem of its own.

I just want to emphasize two points:

1. Using probability distributions to model uncertainty about $$\theta$$
doesn't necessarily mean that the parameter itself has erratic/random behavior.
Most of the time in practice (and everywhere in this post),
what is actually being modeled by
these distributions (over the parameter, yes)
is **my personal knowledge** on the thing rather than the thing itself.
1. Accordingly, once we believe that the combo prior + likelihood reasonably
accomodates reality, Bayes' theorem offers the logical
foundation upon which we should rationally
adjust our perceptions on $$\theta$$ under new evidence.

The [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)
is a good example of that.
Bear in mind when thinking about it that
the player's point of view is the subject of the model,
not some kind of frequentist probability.
It's a matter of incomplete knowledege:
if you were the one to put the prize there,
you would know the answer for
sure (perfect knowledge, zero uncertainty).

## The key idea in Thompson Sampling

Each time an user enters the app,
we randomly choose a version of the app (here, A or B)
and observe either a positive or a negative outcome.
Here I only treat the case of binary outcomes,
but the overall rationale is useful for continuous measurements
as well (e.g. how much money was spent).

If we assume that the outcomes of different users don't depend on each other
and, for simplicity, that each user $$i$$ has only 1 outcome,
the outcomes of each group of users (A and B)
can be seen as samples from Bernoulli distributions with parameters

$$
\theta_A = P(\text{positive outcome} \mid A)
\quad \text{and} \quad
\theta_B = P(\text{positive outcome} \mid B).
$$

Whenever evidence favors $$\theta_B > \theta_A$$,
we want version B to be preferred over A, and vice-versa.
As time goes by and we observe user outcomes,
we gather more and more information on these parameters.

Following the Bayesian approach,
let's denote by $$p_{n,A}(\theta)$$ and $$p_{n,B}(\theta)$$
the distributions containing our knowledge on
$$\theta_A$$ and $$\theta_B$$
after we've learned from the experience of
the $$n \geq 0$$ first users.

The key idea in Thompson Sampling is:
any time you need to choose a variation,

- Compute

$$\begin{align}
P(\theta_B > \theta_A)
&= \int_0^1 p_{n,B}(\theta_B)
\left(
\int_0^{\theta_B} p_{n,A}(\theta_A) d\theta_A
\right) d\theta_B,
\end{align}$$

- and then choose B with probability
$$P(\theta_B > \theta_A)$$.

That's very intuitive:
considering all you know at the time,
sample version B (or A) as often as you believe
that it's more likely to yield a positive outcome.

## Updating priors

Now let's look into how to adapt our priors,
or how to acquire knowledge, if you like.

If we start at $$n=0$$ with no predilections,
then we may assign a uniform (flat) prior for both A and B:

$$
p_{0,A}(\theta) = p_{0,B}(\theta) = 1, \quad \theta \in (0,1).
$$

That means we believe that pretty much anything can happen in both scenarios.
There may be better choices; if you think about it,
that implies very strong assumptions, but follow along for now.

This way, $$P(\theta_B > \theta_A) = 0.5$$ and A and B
are equally likely to be chosen.

Let's say that we saw the experience of the first
$$n = n_A + n_B$$ users and

- $$n_A$$ of them happened to be exposed to A,
with $$s_A$$ positive outcomes, or successes;
- same thing for B, with $$n_B$$ and $$s_B$$.

For each group, these results can be seen as
[Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
experiments, since they are the sum of many Bernoulli experiments.
For A, and analogously for B, the posterior is

$$\begin{align}
p_{n,A}(\theta \mid s_A, n_A)
&\propto \text{likelihood} \times \text{prior} \\
&= P\left(\mathrm{Bin}(n_A, \theta) = s_A\right) \times p_{0,A}(\theta) \\
&= {n_A \choose s_A} \theta^{s_A} (1-\theta)^{n_A - s_A} \times 1 \\
&\propto \theta^{s_A} (1-\theta)^{n_A - s_A}.
\end{align}$$

As we saw earlier, we just need to integrate this
[kernel](https://en.wikipedia.org/wiki/Kernel_(statistics)#Bayesian_statistics)
to get the normalizing constant:

$$
\int_0^1 \theta^{s_A} (1-\theta)^{n_A - s_A} d\theta
= \frac{s_A! (n_A - s_A)!}{(n_A + 1)!},
$$

where the solution comes from iteratively applying
[integration by parts](https://en.wikipedia.org/wiki/Integration_by_parts),
since $$s_A$$ and $$n_A$$ are integers
(more on that
[here](https://en.wikipedia.org/wiki/Rule_of_succession#Mathematical_details)).
The solution of integrals of this form for not-necessarily-integer values
of $$s_A$$ and $$n_A$$ is important in other math problems
and it's known as the
[Beta function](https://en.wikipedia.org/wiki/Beta_function),
$$B(s_A + 1, n_A - s_A + 1)$$ in this case,
resulting in what is known as the
[Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
$$\mathrm{Beta}(\alpha_A, \beta_A)$$ with parameters
$$\alpha_A = s_A + 1$$ (positive outcomes + 1) and
$$\beta_A = n_A - s_A + 1$$ (negative outcomes + 1)
as the posterior:

$$\begin{align}
p_{n,A}(\theta \mid s_A, n_A)
&= \frac{\theta^{\alpha_A - 1} (1-\theta)^{\beta_A - 1}}{B(\alpha_A, \beta_A)} \\
&= \frac{\Gamma(\alpha_A + \beta_A)}{\Gamma(\alpha_A) \Gamma(\beta_A)}
\theta^{\alpha_A - 1} (1-\theta)^{\beta_A - 1},
\end{align}$$

where $$\Gamma$$ is the [Gamma function](https://en.wikipedia.org/wiki/Beta_function),
which can be used to express the Beta function.
Using the fact that $$\Gamma(x) = (x-1)!$$ when $$x$$ is integer,
we see that this is indeed the same normalizing constant as
the one computed in the previous equation.

**What did we learn?**
We saw that the well-known Beta distribution arises *naturally*
(i.e., as a logical necessity) as the bayesian posterior
under a Uniform prior + a Binomial likelihood!
As you may already know, the Uniform is a special case
of the Beta distribution when $$\alpha = \beta = 1$$.
That is: both the prior and the posterior for $$\theta$$
are Beta distributions under the Binomial likelihood
in which $$\theta$$ is the probability of a positive outcome.
In that case, we say that the Beta distribution is a
[conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)
for the Binomial likelihood: when we update it with new data
following the Bayes law, we stay in the same parametric
family of distributions.

If we repeat this process, we obtain the formula that we need
to implement all of this.
Let's say now that
$$m = m_A + m_B$$ additional users came
along with $$0 \leq t_A \leq m_A$$ and $$0 \leq t_B \leq m_B$$ successes.
Now $$p_{n,A}(\theta \mid s_A, n_A)$$ acts as our prior
and the posterior for $$\theta$$ under version A becomes

$$\begin{align}
p_{n+m,A}(\theta \mid s_A, n_A, t_A, m_A)
&\propto P\left(\mathrm{Bin}(m_A, \theta) = t_A\right)
\times p_{n,A}(\theta \mid s_A, n_A) \\
&\propto \theta^{t_A} (1-\theta)^{m_A - t_A}
\times \theta^{s_A} (1-\theta)^{n_A - s_A} \\
&= \theta^{s_A + t_A} (1-\theta)^{(n_A + m_A) - (s_A + t_A)},
\end{align}$$

which is just a Beta distribution with parameters
$$\alpha_A = s_A + t_A + 1$$ and $$\beta_A = (n_A + m_A) - (s_A + t_A) + 1$$.
Notice that's the same as waiting
for the results of all $$n+m$$ users,
given that we observe the same shares between A and B and
the same user outcomes within each version,
which would not be equally likely to occur in case we have updated earlier.
It's hopefully clear that the updates for A and B have the form

$$\begin{align}
\alpha &\leftarrow \alpha + \text{new successes since last update} \\
\beta &\leftarrow \beta + \text{new failures since last update}
\end{align}$$

**Interpretation of the hyper-parameters.** That makes clear that $$\alpha$$ counts
successes and $$\beta$$ counts failures.
Also:

- The mean of the Beta distribution is $$\alpha/(\alpha + \beta)$$;
- The mode, which exists if $$\alpha,\beta > 1$$ (always true after the first update
if we start at the Uniform), is $$(\alpha-1)/(\alpha+\beta-2)$$.

After we've seen sufficiently many users, both the mean and the mode
are very close to the
proportion of positive outcomes (in each version A/B),
which is a nice property.

**Randomization rule.** Finally, one can
[show](https://www.evanmiller.org/bayesian-ab-testing.html#binary_ab_derivation)
that, four our case,

$$
P(\theta_B > \theta_A) =
\sum_{i=0}^{\alpha_B-1} \frac{B(\alpha_A + i, \beta_A + \beta_B)}
{(\beta_B + i)B(1+i, \beta_B)B(\alpha_A, \beta_A)}.
$$

That makes sense because the hyper-parameters are integers
(at least one of the pairs $$\alpha, \beta$$ must be,
see why [here](https://stats.stackexchange.com/a/25297)).
In general, you can always use numerical integration.
Here is an example function in R using the instance
of $$P(\theta_B > \theta_A)$$ introduced before for Beta distributions:

```r
prob_B_is_better = function(alpha_a, beta_a, alpha_b, beta_b) {
  integrate(function(theta) {
    dbeta(theta, alpha_b, beta_b) * pbeta(theta, alpha_a, beta_a)
  }, lower = 0, upper = 1)$value
}
```

I've used the fact that the inner integral is just the
[cumulative distribution
function (CDF)](https://en.wikipedia.org/wiki/Cumulative_distribution_function).

## Simulations

The code to reproduce the analysis in this section is available
[here](https://github.com/abelborges/abelborges.github.io/tree/master/code/thompson-sampling).

Simulations are useful to assess the behavior
of probabilistic methods under as many
hypothetical scenarios as we want.
We're going to focus on the following questions
of practical relevance:

1. When to stop the experiment?
1. How to estimate the lift $$\Delta = \theta_B - \theta_A$$?

Well, I want to stop as soon as I'm
convinced by evidence that $$\Delta$$ is probably not zero.

More specifically,
the knowledge distributions for $$\theta_A$$
and $$\theta_B$$ imply a knowledge distribution
for $$\Delta$$, say $$p_n(\Delta)$$.
Then, at any time, we can deduce
[credible intervals](https://en.wikipedia.org/wiki/Credible_interval)
for $$\Delta$$.
Here, I take the mean as the point estimate.
For the uncertainty estimate, the credible intervals,
the most common options are:

- The so-called equal-tailed interval (ETI)
is defined by the quantiles
$$\alpha/2$$ and $$1-\alpha/2$$
for a $$100\alpha\%$$ credible interval
(e.g. if $$\alpha=80\%$$, then we use percentiles 10 and 90);
- The
[Highest Density Region (HDR)](https://www.webpages.uidaho.edu/~stevel/517/Computing%20and%20Graphing%20HDR.pdf),
the subset of the support of a distribution
that accumulates a specified probability mass,
say $$\alpha$$,
*and* have the highest possible values
of the density function.

The ETI equals the HDR in case the distribution is
symmetric and unimodal.
For skewed distributions, as is our case,
the ETI may let values near the mode out of the interval
while including values with lower associated density,
and I want to avoid that property.
The HDR certainly contains the mode,
which is unique and very close to the mean in our case,
so I'm going to use it.

In any case, the answers for questions 1 and 2
can be merged into a single stopping rule like
*"Stop as soon as the credible interval for $$\Delta$$
doesn't contain 0"*,
or
*"Stop as soon as the width of the credible interval
for $$\Delta$$ becomes less than a threshold"*,
or a combination of these.

The scenarios we may face can be defined essentially
in terms of $$\theta$$.
The cross product of

- The baseline rate of positive outcomes $$\theta_A$$
  being 1%, 5% or 10%; and
- The relative lift $$\Delta/\theta_A$$
  being 0%, 10% or 100%.

amounts to 9 scenarios and these are the ones
I consider here.

**Now to the simulation.**
First, I want to understand how
$$P(\Delta > 0) = P(\theta_B > \theta_A)$$
behaves under different conditions.
The
[generative model](https://en.wikipedia.org/wiki/Generative_model)
for user outcomes and the algorithm
we devised in the previous section put together
look like the following.
We start with uniform priors, i.e.
$$\alpha_A = \beta_A = \alpha_B = \beta_B = 1$$.
Then, for each new user $$n \geq 1$$:

1. *Thompson Sampling*: Compute
   $$P(\theta_B > \theta_A)$$
   and sample the variation based on it;
1. Sample the (binary) user outcome using
   $$\theta_A$$ or $$\theta_B$$,
   depending on the previous step;
1. Update the hyper-parameters of the sampled variation,
   increasing $$\alpha$$ or $$\beta$$ in 1
   depending on whether the experience was
   positive or negative, respectively.

Notice that I'm updating the distributions
right after observing the outcome
of each user. In practice, that may not
be feasible. Updates could be executed
after counting results from batches of users.

I've used a total of 10K users and repeated this whole
process 100 times, resulting in 1M simulated user experiences
for each scenario.
Now, we can visualize, say,
the average (dense line) + 80% ETI (shadows)
of the $$P(\Delta > 0)$$
values across the 100 histories as a function
of the number of users, $$n$$,
for each scenario.

> Amusingly, this is a
[frequentist](https://en.wikipedia.org/wiki/Probability_interpretations#Frequentism)
analysis of subjective probabilities.
Note that these are not bayesian estimates.
The uncertainty measured by these intervals
are with respect to the unobserved, alternative histories
that are possible according to our generative model.
I want to assess in a simple manner how my
perception of the reality at a certain point
in time may deviate from the actual
underlying reality itself.
In a minute, we're going to take a closer look
into the knowledge distribution for $$\Delta$$,
which is the only thing we have in practice
to both (1) measure uncertainty and (2) make decisions.

![]({{site.baseurl}}/images/thompson-sampling/simple-scenarios.png)

- As expected, the greater the lift the easier to catch
the difference sooner;
- For a fixed (relative) lift, it's easier to
detect an existing difference if the rates
are bigger. That's just because the absolute value
of the lift, $$\theta_B - \theta_A$$, is also bigger in such cases.
- Under no lift, it looks like the $$P(\theta_B > \theta_A)$$
time series may drift up and down with no signs of convergence.
Notice the wide confidence intervals irrespectively of $$\theta_A$$ and $$n$$.
- Both the green curve in the first pane and
the blue curve in the third pane correspond to $$\Delta = 0.01$$:
notice the difference in how easy it is to catch the lift.

**Now let's move to the distribution of $$\Delta$$**.
Since $$\Delta = \theta_B - \theta_A$$, we have

$$\begin{align}
p_n(\Delta)
&= \int_0^1 p_{n,B}(\Delta + \theta) p_{n,A}(\theta) d\theta \\
&= \frac{\int_0^1 (\Delta+\theta)^{\alpha_B-1} (1-\Delta-\theta)^{\beta_B-1}
\theta^{\alpha_A-1} (1-\theta)^{\beta_A - 1} d\theta}
{B(\alpha_A, \beta_A) B(\alpha_B, \beta_B)}.
\end{align}$$

You can check the closed form solution to the integral
[here](https://www.terrapub.co.jp/journals/jjss/pdf/4002/40020265.pdf).
It depends on the
[Appell hypergeometric function](https://en.wikipedia.org/wiki/Appell_series)
$$F_3$$, and it's kinda hard to compute it.
Since I didn't find code to do so,
I'll resort to numerically dealing with the above expression.

In the simulations, I record the most up-to-date values of
the hyper-parameters (`universe` counts the 100
replications and `b_is_better` is $$P(\Delta > 0)$$):

```
# A tibble: 9,000,000 x 9
   theta_a theta_b universe nth_user b_is_better alpha_a beta_a alpha_b beta_b
     <dbl>   <dbl>    <int>    <int>       <dbl>   <int>  <int>   <int>  <int>
 1     0.1     0.1        1        1       0.5         1      1       2      1
 2     0.1     0.1        1        2       0.667       1      1       2      2
 3     0.1     0.1        1        3       0.5         1      2       2      2
 4     0.1     0.1        1        4       0.7         1      3       2      2
 5     0.1     0.1        1        5       0.8         1      3       3      2
 6     0.1     0.1        1        6       0.886       1      4       3      2
 7     0.1     0.1        1        7       0.929       1      5       3      2
 8     0.1     0.1        1        8       0.952       1      5       3      3
 9     0.1     0.1        1        9       0.917       1      5       3      4
10     0.1     0.1        1       10       0.879       1      5       4      4
# … with 8,999,990 more rows
```

I discovered
[here](https://stackoverflow.com/a/42987104/6152355)
a piece of code from
[this](https://sites.google.com/site/doingbayesiandataanalysis/software-installation)
book to compute the HDRs for single mode distributions (our case)
given a quantile function (the inverse of the CDF) and $$\alpha$$,
the probability of the credible interval.
The CDF of $$\Delta$$
(remark the support of $$\Delta$$ is $$(-1,1)$$)
is

$$
\Delta \mapsto \int_{-1}^\Delta p_n(t) dt
$$

and since we know it's monotonically non-decreasing
we can easily compute its
[inverse](https://stackoverflow.com/a/10081571/6152355)
on-demand via standard root-finding numerical methods,
like Newton's. Here's some R code to do that:

```r
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
```

## The impact of better priors



## More realistic scenarios

[This](https://medium.com/pinterest-engineering/trapped-in-the-present-how-engagement-bias-in-short-run-experiments-can-blind-you-to-long-run-58b55ad3bda0)
post on the Pinterest engineering blog
hightlight a very important issue they call "engagement bias"

> [...] engagement bias: your treatment doesn’t have the
same effect on unengaged users as it does on engaged users,
but the engaged users are the ones who show up first and
therefore dominate the early experiment results.
If you trust the short-term results without accounting for and
trying to mitigate this bias, you risk being trapped in the present:
building a product for the users you’ve already activated
instead of the users you want to activate in the future.

That's specially undesirable in case you're targeting new users.
How can we take it into account?

The boring solution is just to wait longer so that we can be more confident.
But we have no time for that, we've got many other theories to test.
I'm interested in how we may be able to catch such
behavior early on in the experiment.
Let's accomodate the possibility of engagement bias into
our knowledge model so that we're not fooled by it.
What we are about to do is called
[hierarchical modeling](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling).

The idea is to 





