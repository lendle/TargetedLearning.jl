<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
              processEnvironments: true
             }
  );
</script>

#The observed data structure

In TargetedLearing.jl we are dealing with observed data of the form $O=(W, A, Y)$ where $W$ is a real valued vector, $A$ is binary, and $Y$ is a real number. Typically $W$ will represent baseline covariates, $A$ represents an indicator of treatment (or of missingness), and $Y$ represents an outcome. Currently, we require that $Y$ is binary or bounded between 0 and 1. If that's not the case with your data set, you can easily rescale it with
```julia
newY = (Y .- minimum(Y) ./ (maximum(Y) .- minimum(Y))
```

Assume we observe $n$ [i.i.d.](http://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) copies of $O$:
$$
\\{O_1, O_2, \ldots O_n\\}
$$
where $O_i = (W_i, A_i, Y_i)$ and that a single observation $O$ has distribution $P_0$.

# Counterfactuals and causal parameters

To motivate the statistical estimation problem, let's define counterfactual outcomes. Suppose there exists some value $Y_a$ that is the outcome that would have been observed for a particular observation, had $A$ been $a \in \\{0, 1\\}$. Because $A$ might not have been $a$, we don't necessarily get to observe $Y_a$, and we'll never get to observe both $Y_1$ and $Y_0$ for the same observation, so we call these random variables counterfactual or potential outcomes.

Using counterfactuals, we can define some interesting causal parameters. Examples that can be estimated by this package are listed here. Here, $E_0$ denotes expectation w.r.t. the true distribution of counterfactuals.

* **Counterfactual mean under static regime**: $E_0(Y_a)$ for $a \in \\{0, 1\\}$.
* **Counterfactual mean under dynamic regime**: $E_0(Y_{d(W)})$ where $d$ is a known function of $W$ mapping to $\\{0, 1\\}$.
* **Average treatment effect (ATE)**: $E_0(Y_1) - E_0(Y_0)$ is the typical definition, or $E_0(Y_{d_1(W)})-E_0(Y_{d_0(W)})$ with two dynamic regimes.

Other types of contrasts (*e.g.* ratios) or arbitrary transformations of counterfactual means are not estimated directly, but can be computed given estimated of the counterfacutal means.

## Assumptions

These quantities are functions of the distribution of counterfactual random variables. Under some causal assumptions, these parameters can be written as functions of the distribution of the observed data. (In which case, the causal parameters are said to be identifiable).

For estimating a parameter that depends counterfactual means under regimes in a set $\mathcal D$,
we make the **randomization assumption**
$$
Y_{d(W)} \perp A \mid W \text{ for all } d \in \mathcal {D}
$$
and the **positivity assumption**
$$
g_0(d(W) \mid W) > 0 \text{ a.e. for all } d \in \mathcal{D}.
$$
For a single counterfactual mean, $\mathcal{D}$ contains only one regime. The assumptions are written in terms of dynamic regimes, but a static regime is a special case of a dynamic regime $d(W)$ is constant w.r.t. $W$.

The randomization assumption is sometimes called the "no unmeasured confounders" assumption. It means that any variables that can influence both the treatment and outcome are measured in $W$.  This assumption is not testable, and requires domain area knowledge to determine if it is reasonable or not.

The positivity assumption is also called the experimental treatment assignment assumption. It can be interpreted as requiring that a particular treatment regime of interest has to have some chance of occuring for (almost) every value of $W$.

# Statistical target parameters

There are two main classes of parameters estimated by TargetedLearing.jl., which are specified `param` keyword argument of estimation functions.

The first class of parameters corresponds to a counterfacutal mean (under assumptions):
$$
E_0(E_0(Y\mid A=a, W))
$$
for a static regime
or
$$
E_0(E_0(Y\mid A=d(W), W))
$$
for a dynamic regime.

To specify a mean parameter of this sort, set `param=Mean(a)` where `a` is either a scalar `0.0` or `1.0` for a static regime, or a vector for a dynamic regime. For the latter, `a` should be of length $n$ and the $i$th element should be equal to the treatment the $i$th observation would get under the regime of interest.

The second class of parameters corresponds to the difference in counterfactual means:
$$
E_0(E_0(Y\mid A=1, W) - E_0(Y\mid A=0, W))
$$
or more generally,
$$
E_0(E_0(Y\mid A=d_1(W), W) - E_0(Y\mid d_0(W), W)).
$$

To specify such a parameter, pass `param=ATE(a1, a0)`, where, like above `a1` and `a0` should be scalars for static regimes or vectors for dynamic regimes.
You can specify both static or dynamic regimes, or one of each.
By default `ATE()=ATE(1.0, 0.0)` is the usual ATE.



# Missing data

Missing outcomes and counterfactuals are nearly identical. Suppose $Y$ is missing for some observations, and instead of $A$ being treatment, $A$ is an indicator of whever or not $Y$ is missing.

TODO: more details and an example.
