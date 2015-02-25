<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
              processEnvironments: true
             }
  );
</script>

# TL;DR

* The asymptotic properties of an asymptotically linear (AL) estimator are defined by its influence curve (IC).
* The delta method and continuous mapping theorem give us a way to compute the IC of a transformation of an AL estimator given the original IC. (It just requires taking derivatives and some multiplication).
* Automatic differentiation differentiates automatically
* Combining these, we can automatically compute the IC (and therefore standard errors, confidence intervals and so on) automatically of arbitrary transformations of AL estimators without having to work out the IC by hand.

# Asymptotic Linearity

__Definition__ *An estimator $\psi_n$, (a function of i.i.d. observations $O_1, \ldots O_n$,) for $\psi \in \mathbb{R}^k$ is **asymptotically linear** if
$$
\sqrt{n} (\psi_n - \psi_0) = \frac{1}{\sqrt{n}} \sum_{i=1}^n D(O_i) + o_p(1)
$$
such that $E_0(D)=0$ and $E_0(DD^\intercal)$ is finite and non-singular.* The function $D$ is called the influence curve or influence function. 

By the central limit theorem, we have that $\sqrt{n}\psi_n \leadsto N(\psi_0, E_0(DD^\intercal))$.

# The Delta Method

__Theorem__ *Let $\phi: \mathbb{D}_\phi \subset \mathbb{R}^k \mapsto \mathbb{R}^m$ be differentiable at $\psi$.* 
*Let $\psi_n \in \mathbb{D}_\phi$.*
*If $\sqrt{n}(\psi_n - \psi)$ converges in distribution, then $$\sqrt{n}(\phi(\psi_n) - \phi(\psi)) = \phi'_\psi(\sqrt{n}(\psi_n - \psi)) + o_P(1).$$*
For a proof of a more general result, see Theorem 3.1 in [Asymptotic Statistics by van der Vaart](http://www.cambridge.org/rs/academic/subjects/statistics-probability/statistical-theory-and-methods/asymptotic-statistics?format=PB). A special case of van der Vaart Theorem 3.1 is what [wikipedia](https://en.wikipedia.org/wiki/Delta_method) and Statistics 101 call the delta method, where it is typically assumed that $\sqrt{n}(\psi_n - \psi) \leadsto N(0, \Sigma)$. 

If $\psi_n$ is asymptotically linear with IC $D$, then by the continuous mapping theorem and the definition of asymptotic linearity we have 

\begin{align}
\sqrt{n}(\phi(\psi_n) - \phi_0) =& \phi_{\psi_0}' \sum_{i=1}^n D(O_i) + o_p(1)\\\\
=&  \sum_{i=1}^n \phi_{\psi_0}'D(O_i) + o_p(1)
\end{align}

This means that the IC of the transformed estimator $\phi(\psi_n)$ is $\phi_{\psi_0}D$. Note that if $\phi_{\psi_0}'$ is $0$, this is not very interesting.


# Automatic differentiation

*Automatic differentiation* is weird because if you don't know what it is, you probably think that you do. Automatic differentiation is **not** *symbolic differentiation* or *numerical differentiation*.

*Symbolic differentiation* is when you take an expression, say `sin(x)`, that you want to differentiate, and replace it with an expression for the derivative (`cos(x)`). This can be done by computers, and is what is going on if you ask [WolframAlpha](http://www.wolframalpha.com/input/?i=d+sin%28x%29%2Fdx) for the derivative `sin(x)` w.r.t. `x`.  Symbolic diff is nice and exact but can explode in complexity, and you need an expression composed of parts that the system already knows how to differentiate, not an arbitrary subroutine.

*Numerical differentiation* is when you approximate the derivative of some function at a particular value by evaluating the function at some perturbed values of the input and use linear interpolation. For example, you could approximate `sin'(x)` at `x=5` by computing `(sin(5.0001)-sin(5))/0.0001`.
Numerical diff is easy to implement but is approximate and can be slow, particularly when you need to compute the gradient of a function of many inputs.

*Automatic differentiation* doesn't do either of these things. Basically, it takes some arbitrary subroutine you write as input, and produces a new subroutine that computes the derivative. The produced code computes the exact derivative, not an approximation. In many cases it's essentially the same code that you would write if you were to do it by hand. If the target function is not differentiable everywhere, you'll still get code that computes the derivative correctly where the function is differentiable. It can also be used on arbitrary algorithms, for example

```julia
function f(x::Vector{Float64})
    total = 0.0
    for y in x
        if y > 0.0
            total += foo(x)
        else
            total += bar(x)
        end
    end
    total
end
```
Symbolic differentiation wouldn't know what to do with the for loop and if statement. Autodiff doesn't have a problem with this. 

[Here](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/) is an interesting blog post about autotodiff and another [here](https://justindomke.wordpress.com/2009/11/30/automatic-differentiation-without-compromises/). [This](https://justindomke.wordpress.com/2009/03/24/a-simple-explanation-of-reverse-mode-automatic-differentiation/) post describes one method called reverse-mode automatic differentiation. There are a handful of packages that implement different kinds of autodiff in Julia with information available at [juliadiff.org](http://juliadiff.org).  [www.autodiff.org](http://www.autodiff.org) has a lot of useful information too.

A particularly easy to understand method for autodiff uses [dual numbers](https://en.wikipedia.org/wiki/Dual_number#Differentiation). The [DualNumbers.jl](https://github.com/JuliaDiff/DualNumbers.jl) package.

# Putting it all together

The estimation methods in TargetedLearning.jl returns objects that are subtypes of the `Estimate` type which contain estimated influence curve information. Using operator overloading to implement automatic differentiation in nearly the same way that DualNumbers.jl does, we can compute arbitrary an arbitrary transformation $\phi$ of `Estimate`s and automatically calculate an estimate of the IC of the transformed `Estimate` without having to work it out by hand.  The estimate of the IC of $\phi(\psi_n)$ is computed as $\phi'_{\psi_n} D_n$ where $D_n$ is the estimated IC of $\psi_n$.

For example suppose we have `Estimate` objects `ey1` and `ey0` which are estimates of $E(Y_1)$ and $E(Y_0)$ (mean counterfactual outcomes under treatments 1 and 0) respectively. If the outcome is binary, we might be interested in the log causal [odds ratio](https://en.wikipedia.org/wiki/Odds_ratio). We can compute that as `log((ey1/(1-ey1))/(ey0/(1-ey0)))`. This expression will return a new `Estimate` which includes an estimated influence curve for the log causal odds ratio. 

*It's important to remember that if $\phi_{\psi_0}'$ is $0$, this might not be so useful.*
