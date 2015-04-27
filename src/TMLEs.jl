module TMLEs

using Docile
@document

export tmle

using ..Qmodels, ..Parameters, ..Common

type TMLE{T<:FloatingPoint} <: AbstractScalarEstimate
    psi::T
    ic::Vector{T}
    n::Int
    estimand::String
    function TMLE(psi, ic, n, estimand)
        est = ScalarEstimate(psi, ic)
        new(est.psi, est.ic, n, estimand)
    end
end

TMLE{T<:FloatingPoint}(psi::T, ic::Vector{T}, n, estimand) = TMLE{T}(psi, ic, n, estimand)


"""
Performs targeted minimum loss-based estimation

** Arguments **

* `logitQnA1` - Vector of length n of initial estimates logit(\bar{Q}_n(1, W_i))
* `logitQnA0` - Vector of length n of initial estimates logit(\bar{Q}_n(0, W_i))
* `gn1` - Vector of length n of initial estimats g_n(1, W_i)
* `A` - Vector of length n of treatments, 0.0 or 1.0
* `Y` - Vector of length n of outcomes, bounded between 0 and 1.

** Keyword Arguments **

* `param` - Target parameter. See ?ATE or ?Mean. Default is ATE()
* `weightedfluc` - A boolean, `false` by default.
If `false`, the "standard" fluctuation is performed, there the fluctuation covariate is divided
by \(g_n(A_i\mid W_i)\), and weights are set to 1.
If `true`, the fluctuation covariate is used directly, and weights are set to the reciprical
of \(g_n(A_i\mid W_i)\) where \(g_n(A_i \mid W_i)\) is computed as `gn1[i]` if `A[i]` is 1, or `1-gn1[i]` otherwise.

"""
function tmle(logitQnA1::Vector, logitQnA0::Vector, gn1::Vector, A, Y;
              param::Parameter=ATE(),
              weightedfluc::Bool=false)
    q = Qmodel(logitQnA1, logitQnA0)
    fluctuate!(q, param, gn1, A, Y, weighted=weightedfluc)
    TMLE(applyparam(param, q, A, Y)..., nobs(q), estimand(param))
end

end
