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
at $$x$$ assuming data was already observed.

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
It's just the normalizing constant that makes $$p(\theta \mid x)$$
integrate to 1.
The harshness of obtaining a clear picture of the posterior
in a given setup (likelihood + prior + data) is a problem of its own.

I just want to emphasize two points:

1. Using probability distributions to model uncertainty about $$\theta$$
doesn't necessarily mean that the parameter itself has erratic/random behavior.
Most of the time, what is actually being modeled by
these distributions (over the parameter, yes)
is **my personal knowledge** on the thing rather than the thing itself.
1. Accordingly, once we believe that the combo prior + likelihood reasonably
accomodates reality, Bayes' theorem offers the logical foundation upon which
we rationally adjust our perceptions on $$\theta$$ under new evidence.

The [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)
is a good example of that.
Bear in mind when thinking about it that
the player's point of view is the subject of the model,
not some kind of frequentist probability.
It's a matter of incomplete knowledege:
if you were the one to put the prize there,
you would know the answer for
sure (perfect knowledge, no uncertainty).

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
after we've learned from $$n \geq 0$$ users.

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

- and then sample B with probability $$P(\theta_B > \theta_A)$$,
and A otherwise.

That's very intuitive:
considering all you know at the time,
sample version B as often as you believe
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
that's indeed a very strong assumption, but follow along for now.

This way, $$P(\theta_B > \theta_A) = 0.5$$ and A and B
are equally likely to be chosen.

Let's say that we saw the experience of the first
$$n = n_A + n_B$$ users and

- $$n_A$$ of them happened to be exposed to A,
with $$s_A$$ successes, or positive outcomes;
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
function](https://en.wikipedia.org/wiki/Cumulative_distribution_function).

## Simulations
