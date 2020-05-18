---
layout: post
title: The systemic effectiveness of the locally not-so-much effective
excerpt: Testing
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

This is my attempt to understand and highlight
the effectiveness of simple precautionary measures
in a moment like the one we're living in right now
(the Covid-19 pandemic).
These include the so-called nonpharmaceutical interventions,
such as social distancing and the use of masks.

<!--
As this is a non peer-reviewed, personal blog
written for anyone interested in the subject and
willing to independently judge the proposed rationale,
I see no point in just replicating other models I've seen
online or in papers, so I'll try my own.
-->

Also, I want to make clear the underlying
assumptions and why a parameter is set to
some specific value.

## My motivation for taking a closer look

I saw a tweet saying that a 30% effective mask could
cause a reduction on the rate of infections in the system up to 90%.
First, let me try to draw a simple rationale for that claim by using the
[law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability)
on the number $$n$$ of meetings that can occur
to expand the proportion $$p$$ of infected people:

$$\begin{align}
p
&= P(\text{a random person is infected}) \\
&= \sum_{n \geq 0} P(n \text{ meetings occur})
\end{align}$$

## A model for the disease's spread

I follow the common approach of
[compartmental models](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
evolving in discrete time steps measured in days.
Our hypothetical universe consists of a
population of $$N$$ individuals.

Then for each day $$t$$, there are

So, for each individual $$i$$ and day $$t$$,

1. There's a number of $$O_i(t)$$ opportunities in which $$i$$ can be infected;
1. In each such opportunity, there's a probability
$$P(\text{infection})$$ of infection,
which may depend on the state
of $$i$$ (think masks, age group, health status, etc)
and whether or not

