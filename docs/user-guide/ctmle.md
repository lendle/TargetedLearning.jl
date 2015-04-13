<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
              processEnvironments: true
             }
  );
</script>

#Notation

* $O=(W, A, Y)$: Observed data structure
    - $W$: real valued vector of baseline covariates
    - $A$: binary indicator of treatment (or missingness)
    - $Y$: scalar outcome, binary or bounded by $0$ and $1$.
* $O_i = (W_i, A_i, Y_i)$: $i$th observation for $i = 1, \ldots, n$.
* $\mathcal{M}$: statistical model
* $P\in \mathcal{M}$: a distribution of observed data $O$
* $\Psi$: A statistical parameter mapping from $\mathcal{M}$ to $\mathbb{R}^k$.
* $\psi$: $\Psi(P)$ for some $P$
* $E_P$: expectation w.r.t. distribution $P$.
* $\bar{Q}(a, w)$: $E_P(Y\mid A=a, W=w)$ for some $P$
* $Q_W(w)$: $P(W=w)$, the marginal distribution of $W$.
* $Q$: $(\bar{Q}, Q_W)$
* $g(a \mid w)$ : $P(A=a\mid W=w)$
* Subscript $0$ denotes some quantity for the true distribution, *e.g.* $\bar{Q}_0$ is $\bar{Q}$ where the expectation is taken w.r.t. the trued istribution $P_0$.
* Subscript $n$ denotes an estimate based on $n$ observations, *e.g.* $g_n$ is an estimate of $g_0$.

#TMLE

TMLE is a general framework for constructing regular, asymptotically linear plug-in estimators. Details about how to use the TMLE framework to construct an estimator can be found in the [Targeted Learning](https://www.springer.com/statistics/statistical+theory+and+methods/book/978-1-4419-9781-4) book by van der Laan and Rose, or [articles](http://scholar.google.com/scholar?q=targeted+estimation+tmle) on TMLE.

Here we'll walk through an example of how to actually implement a TMLE for the average treatment effect (ATE) and skip over the details of the derivation.

Using [counterfactuals](estimation.md#counterfactuals-and-causal-parameters), we define the ATE as $E_0(Y_1 - Y_0)$. Under causal assumptions, we can write this causal parameter as a parameter of the distribution of the observed data:
\begin{align}
\psi_0 = \Psi(P_0) =& E_0(E_0(Y\mid A = 1, W) - E_0(Y\mid A=0, W)) \\\\
=& E_{Q_{W0}}(\bar{Q}_0(1, W) - \bar{Q}_0(0, W)).
\end{align}
The parameter is computed by first taking the difference $\bar{Q}_0(1, W) - \bar{Q}_0(0, W)$, then averaging over $W$, so $\Psi$ only depends on $P$ through $Q=(\bar{Q}, Q_W)$. Recognizing the abuse of notation, we sometimes write $\Psi(\bar{Q}, Q_W)$. $g_0(a \mid w)$, sometimes called the propensity score, is a nuisance parameter.

A **plug-in** estimator is one that first estimates (relevant parts of) the distribution $P_0$ and then plugs it in to the parameter mapping. In our case, that's $\bar{Q}_0$ and $Q _ {W0}$.
A **TMLE** is a plug in estimator constructed by first finding an initial estimate of $Q_0$, and then updating those estimates using estimates of nuisance parameters ($g _ 0$ here), then computing a plug-in estimate based on the updated estimate of $Q$.
The update is done by fluctuating the initial estimator in such a way that the final plug in estimator is locally efficient and doubly robust.
It turns out in this example, that only the initial estimator of $\bar Q _ 0$ (and not $Q _ {Wn})$ requires a fluctuation.

$\bar{Q} _ 0$ is the conditional mean of $Y$, and can be estimated with regression techniques, _e.g._ logistic regression or something more flexible.  $Q _ {W0}$ is just a marginal distribution, and the easiest way to estimate that is with empirical distribution, so we'll use that for for $Q _ {Wn}$.
The fluctuation relies on an estimate of $g_0$, which is another conditional mean.

<!---

Given estimates $\bar{Q} _ n$ and $Q _ {Wn}$ a simple plug in estimator for $\psi_0$ is be computed as
$$
\Psi(\bar{Q}_n, Q _ {Wn}) = \frac 1n \sum _ {i=1}^n (\bar{Q}_n(1, W_i) - \bar{Q}_n(0, W_i)).
$$

In general, an estimation method may not always result in an estimate that falls in the parameter space, particularly in small samples, even if that method is consistent or even efficient.
Plug-in estimators are one way to guarantee that estimates are always in the parameter space. For example, we know that the ATE must be between $-1$ and $1$, because $Y \in \[0, 1\]$. The plug-in estimator $\Psi(\bar{Q}_n, Q _ {Wn})$ will always be in $\[0, 1\]$, provided $\bar{Q}_n(a, w)$ is yields estimates in $\[0, 1\]$.

Plug-in estimators, however, are not efficient in general. TMLE constructs an efficient plug-in estimator by taking an initial estimate of $Q$, and updates it to $Q_n^*$ in such a way that the so-called efficient influence curve (EIC) equation is solved. For this particular example, only $\bar{Q}_n$ needs to be updated, and not $Q _ {Wn}$.  This update is also called a fluctuation.
 -->

## Fluctuating $\bar{Q}_n$

Fluctuating the initial estimate $\bar{Q} _ n$ amounts to regressing $Y$ on a particular covariate using $\bar{Q} _ n$ as an offset.
In particular, we use the model
<!-- We have a choice of two fluctuation procedures: an *unweighted fluctuation* or a *weighted fluctuation*.
To describe them, we first we define a parametric submodel through the initial $\bar{Q}_n$, $\\{\bar{Q}_n(\epsilon) : \epsilon \in \mathbb{R} \\}$ using an estimate of $g_0$ such that $\bar{Q}_n(\epsilon=0) = \bar{Q}_n$. For the ATE, we will use
 -->
 $$
\mbox{logit} \bar{Q}_n(\epsilon)(a, w) = \mbox{logit} \bar{Q}_n(a, w) + h(a, w) \epsilon
$$
where $\mbox{logit}(x) = \log(x)/\log(1-x)$ and $h$ is a known function.
We also define a loss function for $\bar{Q}_n(\epsilon)$:
$$
\ell(\bar{Q}_n(\epsilon))(O) = - b(A, W) [Y \log(\bar{Q}_n(\epsilon)(A, W)) - (1-Y) \log (1-\bar{Q}_n(\epsilon)(A, W))],
$$
where $b$ is a known function.

If $Y$ is binary, you might recognize this as a logistic regression model for the conditional mean of $Y$ on the covariate $h(A,W)$ with $\mbox{logit} \bar{Q}_n(A, W)$ as a fixed offset and weights $b(A,W)$.
When $Y$ is not binary but bounded between $0$ and $1$, this is still a valid loss function for the conditional mean of $Y$.

In TargetedLearning.jl, there are two types of fluctuations supported, *unweighted* and *weighted*, which define the covariate and weight functions
$h$ and $b$.

* *Unweighted fluctuation*:
\begin{gather}
h(a,w) = \frac{2a-1}{g_n(a\mid w)},  & &
b(a,w) = 1
\end{gather}
* *Weighted fluctuation*:
\begin{gather}
h(a,w) = 2a-1, & &
b(a,w) = \frac{1}{g_n(a\mid w)}
\end{gather}
<!--
Both are chosen such that
$$
h(a,w)b(a,w) = \frac{2a-1}{g_n(a\mid w)}.
This means that the score equation for $\epsilon$ in our quasi-logistic regression model
$$
\frac 1n \sum _ {i=1}^n \frac{2A-1}{g_n(A\mid W)} (Y - \bar{Q}_n(A,W)(\epsilon))
$$
is solved at $\epsilon_n$.
$$ -->

Once this choice is made, we compute an estimate of $\epsilon$ by minimizing the empirical mean of $\ell$:
$$
\epsilon_n = \arg\min _ {\epsilon} \sum _ {i =1} ^n \ell(\bar{Q}_n(\epsilon))(O_i).
$$
This amounts to fitting a (quasi-)logistic regression model with an offset and possibly with a weight.

Finally we set $\bar{Q}_ n ^\* = \bar{Q}_n(\epsilon_n)$ and compute the final estimate of $\psi_0$ as $\Psi(\bar{Q} _ n ^\*, Q _ {Qn})$:
$$
\Psi(\bar{Q} _ n^\*, Q _ {Wn}) = \frac 1n \sum _ {i=1}^n (\bar{Q}\_n^*(1, W_i) - \bar{Q}\_n^\*(0, W_i)).
$$

This final TMLE is a plug-in estimator by definition, and it is also doubly robust and locally efficient, meaning that if either the initial $\bar{Q}_n$ or $g_n$ estimate $\bar{Q}_0$ or $g_0$ consistently respectively, then $\Psi(Q_n^\*)$ is consistent, and if both $\bar{Q}_0$ and $g_0$ are consistent, then $\Psi(Q_n^\*)$ is efficient.

## Example implementation

The math makes things a lot more complicated than they really are. Here's a simple implementation using the unweighted fluctuation:
```julia
# input:
# logitQnA1, logitQnA0: vectors containing initial estimates \logit(\bar{Q}_n(a, W)) for a = 1 and 0, respectively
# gn1: a vector of containing estimates g_n(1 \mid W)
# A: binary treatment vector
# Y: outcome vector
# lreg is a simple wrapper function for logistic regression using the GLMNet.jl package

function simpletmle(logitQnA1, logitQnA0, gn1, A, Y)
    logitQnAA = ifelse(A.==1, logitQnA1, logitQnA0)
    gnA = ifelse(A .==1, gn1, 1 .- gn1)

    Qfit = lreg((2A .-1)./(gnA), Y, offset = logitQnAA)

    Qn⋆A1 = predict(Qfit, 1./gn1, offset = logitQnA1)
    Qn⋆A0 = predict(Qfit, -1./(1 .- gn1), offset = logitQnA0)

    return mean(Qn⋆A1 .- Qn⋆A0)
end
```

## Example using TargetedLearning.jl

Continuing the [example](julia.md#example) from the intro to Julia, we now demonstrate how to use TMLE to estimate the ATE from the Lalonde data set.

```julia
# W: matrix of covariates
# treat: treatment variable
# re78: outcome

tmle(logitQnA1, logitQnA0, gn1, treat, re78, param=ATE(), weightedfluc=false)
```

The `tmle` function returns an estimate along with an estimated variance based on the EIC. It's important to note that this variance is asymptotically consistent if both $\bar{Q}_n$ and $g_n$ are consistent, and is conservative if $g_n$ is consistent but $\bar{Q}_n$ is not. If $g_n$ is not consistent, then EIC based variance estimate is not reliable.

# CTMLE

