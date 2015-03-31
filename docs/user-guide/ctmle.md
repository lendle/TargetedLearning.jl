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
* $g(a \mid w)$ : $P(A=a\mid W=w)$
* Subscript $0$ denotes some quantity for the true distribution, *e.g.* $\bar{Q}_0$ is $\bar{Q}$ where the expectation is taken w.r.t. the trued istribution $P_0$.
* Subscript $n$ denotes an estimate based on $n$ observations, *e.g.* $g_n$ is an estimate of $g_0$.

#TMLE

TMLE is a general framework for constructing regular, asymptotically linear plug-in estimators. Details about how to use the TMLE framework to construct an estimator can be found in the [Targeted Learning](https://www.springer.com/statistics/statistical+theory+and+methods/book/978-1-4419-9781-4) book by van der Laan and Rose, or [articles](http://scholar.google.com/scholar?q=targeted+estimation+tmle) on TMLE.

Here we'll walk through an example of how to actually implement a TMLE for the average treatment effect (ATE) and skip over the details of the derivation.  Using [counterfactuals](estimation.md#counterfactuals-and-causal-parameters), we define the ATE as $E_0(Y_1 - Y_0)$. Under assumptions, we can write this causal parameter as a parameter of the distribution of the observed data:
\begin{align}
\psi_0 = \Psi(P_0) =& E_0(E_0(Y\mid A = 1, W) - E_0(Y\mid A=0, W)) \\\\
=& E_{Q_{W0}}(\bar{Q}(1, W) - \bar{Q}(0, W)).
\end{align}
We see that $\Psi$ can be written in a way that only depends on $P$ through $Q=(\bar{Q}, Q_W)$, so recognizing the abuse of notation, we sometimes write $\Psi(Q)$ or $\Psi(\bar{Q}, Q_W)$.

A **plug-in** estimator is one that first estimates (relevant parts of) the distribution $P_0$ and then plugs it in to the parameter mapping: $\Psi(P_n)$. 
Given an estimate $\bar{Q}_n$
for $\bar{Q}_0$ and letting $Q _ {Wn}$ be the empirical distribution of $W$, a plug in estimator for $\psi_0$ can be computed as
\begin{align}
\Psi(\bar{Q}_n, Q _ {Wn}) =& \int \bar{Q}_n(1, W) - \bar{Q}_n(0, W) dQ _ {Wn} \\\\
 = & \frac 1n \sum _ {i=1}^n (\bar{Q}_n(1, W_i) - \bar{Q}_n(0, W_i)).
\end{align}

In general, an estimation method may not always result in an estimate that falls in the parameter space, particularly in small samples, even if that method is consistent or even efficient.
Plug-in estimators are one way to guarantee that estimates are always in the parameter space. For example, we know that the ATE must be between $-1$ and $1$, because $Y \in \[0, 1\]$. The plug-in estimator $\Psi(\bar{Q}_n, Q _ {Wn})$ will always be in $\[0, 1\]$, provided $\bar{Q}_n(a, w)$ is yields estimates in $\[0, 1\]$.

Plug-in estimators, however, are not efficient in general. TMLE constructs an efficient plug-in estimator by taking an initial estimate of $Q$, and updates it to $Q_n^*$ in such a way that the so-called efficient influence curve (EIC) equation is solved. For this particular example, only $\bar{Q}_n$ needs to be updated, and not $Q _ {Wn}$.  This update is also called a fluctuation.

## Fluctuation

We have a choice of two fluctuation procedures: an *unweighted fluctuation* or a *weighted fluctuation*. To describe them, we first we define a parametric submodel through the initial $\bar{Q}_n$, $\\{\bar{Q}_n(\epsilon) : \epsilon \in \mathbb{R} \\}$ using an estimate of $g_0$ such that $\bar{Q}_n(\epsilon=0) = \bar{Q}_n$. For the ATE, we will use
$$
\mbox{logit} \bar{Q}_n(\epsilon)(a, w) = \mbox{logit} \bar{Q}_n(a, w) + h(a, w) \epsilon
$$
where $\mbox{logit}(x) = \log(x)/\log(1-x)$ where $h(a,w)$ is either $(2a-1)/g_n(a\mid w)$ for the *unweighted fluctuation* or  $2a-1$ for the *weighted fluctuation*.
If $Y$ is binary, you might recognize this as a logistic regression model for the conditional mean of $Y$ on the covariate $h(A,W)$ with $\mbox{logit} \bar{Q}_n(A, W)$ as a fixed offset.

We also define a loss function for $\bar{Q}_n(\epsilon)$: 
$$
\ell(\bar{Q}_n(\epsilon))(O) = - b(A, W) (Y \log(\bar{Q}_n(\epsilon)(A, W)) - (1-Y) \log (1-\bar{Q}_n(\epsilon)(A, W))),
$$
which, when $Y$ is binary, is the negative log likelihood loss function from logistic regression with weight $b(A,W)$. When $Y$ is not binary but bounded between $0$ and $1$, this is still a valid loss function for the conditional mean of $Y$.
For the *unweighted fluctuation*, we choose $b(a,w)$ $1$, and for the *weighted fluctuation*, we choose $b(a,w)$ to be $1/g_n(a \mid w)$.

To fluctuate $\bar{Q}_n$, we compute an estimate of $\epsilon$ by minimizing the empirical mean of $\ell$:
$$
\epsilon_n = \arg\min _ {\epsilon} \sum _ {i =1} ^n \ell(\bar{Q}_n(\epsilon))(O_i).
$$
This amounts to fitting a logistic regression model with an offset and possibly with a weight. 

The covariate and weight functions $h$ and $b$ are chosen such that 
$$
h(a,w)b(a,w) = \frac{2a-1}{g_n(a\mid w)}.
$$
This means that the score equation for $\epsilon$ in our quasi-logistic regression model
$$
\frac 1n \sum _ {i=1}^n \frac{2A-1}{g_n(A\mid W)} (Y - \bar{Q}_n(A,W)(\epsilon))
$$
is solved at $\epsilon_n$.
This is an important part of ensuring that the EIC equation is solved by our plug-in estimator.


The we then set $Q_n^* = (\bar{Q}_n(\epsilon_n), Q _ {Wn})$ and compute the final estimate of $\psi_0$ as $\Psi(Q_n^\*)$. This final TMLE is a plug-in estimator by definition, and it is also doubly robust and locally efficient, meaning that if either the initial $\bar{Q}_n$ or $g_n$ estimate $\bar{Q}_0$ or $g_0$ consistently respectively, then $\Psi(Q_n^\*)$ is consistent, and if both $\bar{Q}_0$ and $g_0$ are consistent, then $\Psi(Q_n^\*)$ is efficient.

## Example implementation 

The math makes things a lot more complicated than they really are. Here's a simple implementation using the unweighted fluctuation:
```julia
# input: 
# logitQnA1, logitQnA0: vectors containing initial estimates \logit(\bar{Q}_n(a, W)) for a = 1 and 0, respectively
# gn1: a vector of containing estimates g_n(1 \mid W)
# A: a vector of 0.0s and 1.0s representing the treatment for each observation
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

Continuing [example](julia.md#example) from the intro to Julia, we now demonstrate how to use TMLE to estimate the ATE from the Lalonde data set.

```julia
# W: matrix of covariates
# treat: treatment variable
# re78: outcome 

tmle(logitQnA1, logitQnA0, gn1, treat, re78, param=ATE(), weightedfluc=false)
```

The `tmle` function returns an estimate along with an estimated variance based on the EIC. It's important to note that this variance is asymptotically valid if both $\bar{Q}_n$ and $g_n$ are consistent, and is conservative if $g_n$ is consistent but $\bar{Q}_n$ is not. If $g_n$ is not consistent, then EIC based variance estimate is not reliable.
