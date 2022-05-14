---
layout: post
title: "Introduction to online experiments with Thompson Sampling"
excerpt: >
  You're going to release a new feature in a product.
  It consists of a modified version B of an existing component A.
  Version B is technically OK but you want to check whether users prefer it over A
  and, if so, gradually point new requests to it.
  This post is about a solution to this problem.
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

You're going to release a new feature in a web application.
It consists of a modified version B of an existing component A.
Version B is technically OK but you want to check whether users prefer it over A
and, if so, gradually point new requests to it.
This post is about a solution to this problem.

One way to check that is by carrying an [A/B test]({{site.baseurl}}/hypothesis-testing).
I want to talk about Thompson Sampling, a Bayesian approach to evolve
our impressions on A and B in an online fashion.

## Bayesian inference

Let's say we're interested in a random process $$X$$
whose realizations are supposed to follow a probability
distribution $$p(x \mid \theta)$$,
parametrized by a vector $$\theta$$
that takes values from a set $$\Theta$$ (of, say, real vectors).
The function $$\theta \mapsto p(x \mid \theta)$$
is called the **likelihood** of $$\theta$$.
It's just another name for the density/probability mass of $$x$$
seen as a function of $$\theta$$, given an observation $$x$$ of $$X$$.

In this setup, inferences about the behavior of $$X$$
(e.g. which values are most likely to occur,
how frequently the values lie on a given interval etc)
can be translated into inferences about $$\theta$$,
a constant whose value we don't know for sure.

Simply put, the Bayesian approach for inference is

1. To model our prior perceptions of what the value of
$$\theta$$ may be with a probability distribution,
say $$p(\theta)$$, known as the **prior**;
1. Then, once we have data, use the
[Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
to deduce the **posterior** distribution of $$\theta$$, namely

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

<!-- The [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)
is a good example of that.
Bear in mind when thinking about it that
the player's point of view is the subject of the model,
not some kind of frequentist probability.
It's a matter of incomplete knowledege:
if you were the one to put the prize there,
you would know the answer for
sure (perfect knowledge, zero uncertainty). -->

## The key idea in Thompson Sampling

Each time an user enters the app,
we randomly choose between A and B
and observe either a positive or a negative outcome.
Here I only treat the case of binary outcomes,
but the overall rationale is useful for continuous measurements
as well (e.g. how much money was spent).

If we assume that the outcomes of different users don't depend on each other
and, for simplicity, that each user $$i$$ has only 1 outcome,
the outcomes of each group of users
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

Let's denote by $$p_{n,A}(\theta)$$ and $$p_{n,B}(\theta)$$
the distributions containing our knowledge on
$$\theta_A$$ and $$\theta_B$$
after we've learned from the experience of
the $$n \geq 0$$ first visitors.

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

Now let's look into how to set and adapt our priors,
or how to acquire knowledge, if you like.

**The prior.**
If we start at $$n=0$$ with no predilections,
then we may assign a uniform prior for A and B:

$$
p_{0,A}(\theta) = p_{0,B}(\theta) = 1, \quad \theta \in (0,1).
$$

That means we believe that pretty much anything
can happen in both scenarios.
In practice, we may be more conservative and assign
a higher probability for A at the beginning.
But let's move on for now.

This way, we start with $$P(\theta_B > \theta_A) = 0.5$$,
thus A and B are equally likely to be chosen.

Let's say that we saw the experience of the first
$$n = n_A + n_B$$ users, where
$$n_A$$ of them happened to be exposed to A,
with $$s_A$$ positive outcomes, or successes;
same thing for B with $$n_B$$ and $$s_B$$.

**The posterior.**
For each group, these results can be seen as
[Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
experiments, since they are the sum of many independent
Bernoulli experiments.
For A, and analogously for B, the posterior is

$$\begin{align}
p_{n,A}(\theta \mid s_A, n_A)
&\propto \text{likelihood} \times \text{prior} \\
&= P\left\{\mathrm{Bin}(n_A, \theta) = s_A\right\} \times p_{0,A}(\theta) \\
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
&= \frac{\theta^{\alpha_A - 1} (1-\theta)^{\beta_A - 1}}{B(\alpha_A, \beta_A)}.
\end{align}$$

<!-- &= \frac{\Gamma(\alpha_A + \beta_A)}{\Gamma(\alpha_A) \Gamma(\beta_A)}
\theta^{\alpha_A - 1} (1-\theta)^{\beta_A - 1},
 -->

<!-- where $$\Gamma$$ is the [Gamma function](https://en.wikipedia.org/wiki/Beta_function),
which can be used to express the Beta function.
Using the fact that $$\Gamma(x) = (x-1)!$$ when $$x$$ is integer,
we see that this is indeed the same normalizing constant as
the one computed in the previous equation. -->

**What did we learn?**
We saw that the well-known Beta distribution arises *naturally*
(i.e., as a logical necessity) as the bayesian posterior
under a Uniform prior + a Binomial likelihood.
As you may already know, the Uniform is a special case
of the Beta distribution when $$\alpha = \beta = 1$$.
That is: both the prior and the posterior for $$\theta$$
are Beta distributions under the Binomial likelihood
with $$\theta$$ being the probability of a positive outcome.
In that case, we say that the Beta distribution is a
[conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)
for the Binomial likelihood: when we "Bayes-update" it with new data,
we stay in the same parametric family of distributions.

**Continuously updating.**
If we repeat this process, we obtain a formula
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
&= \theta^{s_A + t_A} (1-\theta)^{(n_A + m_A) - (s_A + t_A)}.
\end{align}$$

We know that this is just a Beta distribution with parameters
$$\alpha_A = s_A + t_A + 1$$ and $$\beta_A = (n_A + m_A) - (s_A + t_A) + 1$$.
Note that the mechanics of the formulae for two updates
is equivalent to performing a single update
after waiting for the results of all $$n+m$$ users.
It's hopefully clear that the updates for A and B have the form

$$\begin{align}
\alpha &\leftarrow \alpha + \text{new successes since last update} \\
\beta &\leftarrow \beta + \text{new failures since last update}
\end{align}$$

**Interpretation of the hyper-parameters.**
That makes clear that $$\alpha$$ counts
successes and $$\beta$$ counts failures.
Also, remember that

- The mean of the Beta distribution is $$\alpha/(\alpha + \beta)$$;
- The mode, which exists if $$\alpha,\beta > 1$$ (always true after the first update
if we start at the Uniform), is $$(\alpha-1)/(\alpha+\beta-2)$$.

Note that after the first tens of users,
both the mean and the mode become very close to the
proportion of positive outcomes observed so far.
Easy to reason about.

**Randomization rule.** Finally,
[one can show](https://www.evanmiller.org/bayesian-ab-testing.html#binary_ab_derivation)
that, for the Beta case,

$$
P(\theta_B > \theta_A) =
\sum_{i=0}^{\alpha_B-1} \frac{B(\alpha_A + i, \beta_A + \beta_B)}
{(\beta_B + i)B(1+i, \beta_B)B(\alpha_A, \beta_A)}.
$$

That makes sense because the hyper-parameters are integers
(at least one of the pairs $$\alpha, \beta$$ must be,
see why [here](https://stats.stackexchange.com/a/25297)).
In general, you can always use numerical integration.
Here is an example function in R using the expression
for $$P(\theta_B > \theta_A)$$ introduced before for Beta distributions:

```r
prob_B_is_better = function(alpha_a, beta_a, alpha_b, beta_b) {
  integrand = function(theta) {
    dbeta(theta, alpha_b, beta_b) * pbeta(theta, alpha_a, beta_a)
  }
  integrate(integrand, lower = 0, upper = 1)$value
}
```

I've used the fact that the inner integral is just the
[cumulative distribution
function (CDF)](https://en.wikipedia.org/wiki/Cumulative_distribution_function).

## Simulate!

Simulations are useful to assess the behavior
of probabilistic methods under as many
hypothetical scenarios as we want.
The code to reproduce the analysis in this section is available
[here](https://github.com/abelborges/abelborges.github.io/tree/master/code/thompson-sampling).

<!-- We're going to focus on the following question of practical relevance:

1. When to stop the experiment?
1. How to provide an estimate of
the "lift" $$\Delta = \theta_B - \theta_A$$?

They are related because intuition suggests
that we should stop as soon as there is "enough evidence"
in favor of $$\Delta$$ being different than zero.
 -->

**How to estimate $$\Delta = \theta_B - \theta_A$$?**
The answer to this question easily translates into
rules for when to stop the experiment.

The knowledge distributions for $$\theta_A$$
and $$\theta_B$$ imply a knowledge distribution
for $$\Delta$$, say $$p_n(\Delta)$$.
Then, at any time, we can compute
properties of the distribution of $$\Delta$$.
For a point estimate, I'll use the mean here.
For uncertainty estimates, the most common options for
[credible intervals](https://en.wikipedia.org/wiki/Credible_interval)
are:

- The so-called equal-tailed interval (ETI)
is defined by the quantiles
$$c/2$$ and $$1-c/2$$
for a $$100c\%$$ credible interval
(e.g. if $$c=80\%$$, then we use percentiles 10 and 90);
- The
[Highest Density Region (HDR)](https://www.webpages.uidaho.edu/~stevel/517/Computing%20and%20Graphing%20HDR.pdf),
the subset of the support of a distribution
that accumulates a specified probability mass,
say $$c$$,
*and* have the highest possible values
of the density function.

The ETI equals the HDR in case the distribution is
symmetric and unimodal.
For skewed distributions, as is our case,
the ETI may let values near the mode out of the interval
while including values with lower probability density,
and I want to avoid that property.
The HDR certainly contains the mode,
which is unique and very close to the mean in our case,
so I'm going to use it.

Given a credibility level $$c$$,
examples of stopping rules based on $$$\Delta$ include:

1. Stop as soon as the credible interval for $$\Delta$$
doesn't contain 0;
1. Stop as soon as the width ("margin of error", if you like)
of the credible interval
for $$\Delta$$ becomes less than a threshold;
1. Stop as soon as the probability of $$\Delta$$
changing sign at some future moment becomes low enough.

I don't know whether the third option is
used in practice, but I find it the most compelling one.
However, it does not seem easy to compute, and requires
some hypotheses about the underlying phenomenon
Maybe we get back to that another time.
Let's go with the first one.
<!-- It's simpler to choose from the first two,
or some combination of them.
 -->

The scenarios we may face can be defined in terms of $$\theta$$.
I consider two dimensions:

- The baseline rate of positive outcomes $$\theta_A$$
  being 1%, 5% or 10%; and
- The relative lift $$\Delta/\theta_A$$
  being 0%, 10% or 100%,

which amounts to 9 scenarios.

**The simulation algorithm.**
The goal here is to get a basic understanding of how

$$P(\Delta > 0) = P(\theta_B > \theta_A)$$

evolves under different conditions.
The
[generative model](https://en.wikipedia.org/wiki/Generative_model)
for user outcomes and the algorithm
we devised in the previous section put together
look like the following.

First, set the "true" values of $$\theta_A$$
and $$\theta_B$$.
We start with uniform priors, i.e.
$$\alpha_A = \beta_A = \alpha_B = \beta_B = 1$$.
Then, for each new user $$n \geq 1$$:

1. *Thompson Sampling*: Compute
   $$P(\theta_B > \theta_A)$$
   and use it to sample a variation;
1. Sample the (binary) user outcome using
   $$\theta_A$$ or $$\theta_B$$,
   depending on the previous step;
1. Update the hyper-parameters of the sampled variation,
   increasing either $$\alpha$$ or $$\beta$$ in 1
   depending on whether the experience was
   positive or negative, respectively.

I'm updating the distributions
right after observing the outcome
of each user. In practice, that may not
be feasible. Updates could be executed
after summarizing results from batches of users.

I've used a total of 10K users and repeated this whole
process 100 times, resulting in 1M simulated user experiences
for each scenario.
With the result, let's look at
the average (dense line) + 80% ETI (shadows)
of the $$P(\Delta > 0)$$
values across the 100 "histories" as a function
of the number of users, $$n$$,
for each scenario.

![]({{site.baseurl}}/images/thompson-sampling/simple-scenarios.png)

- Under no lift (in grey), $$P(\theta_B > \theta_A)$$
drifts up and down with no signs of convergence, which makes sense.
Notice the wide confidence intervals irrespectively of
$$\theta_A$$ and $$n$$;
- As expected, the greater the lift the easier to catch
the difference (if any) sooner;
- For a given relative lift, it's easier to
detect an existing difference if the $$\theta_A$$ is bigger.
But that's just because the absolute value
of the lift is also bigger in such case.
For instance, both the green
curve in the first pane and
the blue curve in the third pane correspond to $$\Delta = 0.01$$.

Amusingly, this is a
[frequentist](https://en.wikipedia.org/wiki/Probability_interpretations#Frequentism)
analysis of subjective ("bayesian") probabilities.
<!-- Note that these are not bayesian estimates. -->
The uncertainty measured by these intervals
are with respect to the
[ensemble](https://en.wikipedia.org/wiki/Statistical_ensemble_(mathematical_physics))
of alternative histories
that are possible according to the combination of the
assumed generative model (what nature yields)
and chosen experiment strategy (our reaction to it).
I wanted to assess in a simple manner how my
perception of the reality at a certain point
in time may deviate from the actual
underlying reality itself.
<!-- In other words, I don't want to be fooled by randomness. -->
In practice, the knowledge distribution for $$\Delta$$
is the only thing we have to
measure uncertainty and make decisions accordingly.

**A closer look into the distribution of $$\Delta$$**.
We want to deduce HDRs for $$\Delta$$,
use them to decide when to stop and then measure the
quality of such decisions under our simple model,
in which $$\theta_A$$ and $$\theta_B$$ are constants.
Since $$\Delta = \theta_B - \theta_A$$,

$$\begin{align}
p_n(\Delta)
&= \int_0^1 p_{n,B}(\Delta + \theta) p_{n,A}(\theta) d\theta
\end{align}$$

whose solution can be found in Theorem 1 of
[this](https://www.terrapub.co.jp/journals/jjss/pdf/4002/40020265.pdf)
paper.
It depends on the
[Appell hypergeometric function](https://en.wikipedia.org/wiki/Appell_series)
$$F_3$$.
[Here](http://mpmath.org/doc/current/functions/hypergeometric.html#appellf3)
you can find the docs of $$F_3$$ in the mpmath Python library,
and below is an implementation of $$p_n(\Delta)$$ making use of it:

```python
from mpmath import beta, mp, appellf3
mp.dps = 25

def delta_cdf(delta, alpha_a, beta_a, alpha_b, beta_b):
    if delta < -1 or delta > 1:
        return None
    
    if abs(delta) < 1e-15:
        f3 = 1
        num = beta(alpha_b + alpha_a - 1, beta_b + beta_a - 1)
    elif delta > 0:
        f3 = appellf3(alpha_a, beta_b, 1 - beta_a, 1 - alpha_b,
                      alpha_a + beta_b, 1 - delta, 1 - delta)
        num = beta(alpha_a, beta_b) * pow(1 - delta, alpha_a + beta_b - 1)
    else:
        f3 = appellf3(alpha_b, beta_a, 1 - beta_b, 1 - alpha_a,
                      alpha_b + beta_a, 1 + delta, 1 + delta)
        num = beta(alpha_b, beta_a) * pow(1 + delta, alpha_b + beta_a - 1) 
    
    den = beta(alpha_b, beta_b) * beta(alpha_a, beta_a)
    return float(f3) * float(num) / float(den)
```

<!-- I discovered
[here](https://stackoverflow.com/a/42987104/6152355)
a piece of code from
[this](https://sites.google.com/site/doingbayesiandataanalysis/software-installation)
book to compute the HDRs for single mode distributions (our case)
given a quantile function (the inverse of the CDF) and $$c$$,
the probability accumulated by the credible interval.
The CDF of $$\Delta$$
(remark the support of $$\Delta$$ is $$(-1,1)$$)
is

$$
\Delta \mapsto \int_{-1}^\Delta p_n(t) dt
$$

and since we know it's non-decreasing
we can easily compute its
[inverse](https://en.wikipedia.org/wiki/Inverse_function#Two-sided_inverses)
on-demand via standard root-finding numerical methods,
like Newton's. Here's some [R code](https://stackoverflow.com/a/10081571/6152355) to do that: -->

<!-- 
```r
delta_pdf = function(d, alpha_a, beta_a, alpha_b, beta_b) {
  integrate(
    function(t) dbeta(d + t, alpha_b, beta_b) * dbeta(t, alpha_a, beta_a),
    lower = 0,
    upper = 1
  )$value
}

delta_cdf = function(d, alpha_a, beta_a, alpha_b, beta_b) {
  integrate(
    Vectorize(delta_pdf),
    lower = -1,
    upper = d,
    alpha_a = alpha_a,
    beta_a  = beta_a,
    alpha_b = alpha_b,
    beta_b  = beta_b
  )$value
}

delta_icdf = function(p, alpha_a, beta_a, alpha_b, beta_b) {
  uniroot(
    function(d) p - delta_cdf(d, alpha_a, beta_a, alpha_b, beta_b),
    interval = c(-1, 1)
  )$root
}
```
 -->

**Measuring the quality of stopping rules.**
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

