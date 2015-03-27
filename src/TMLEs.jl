module TMLEs

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

function tmle(logitQnA1::Vector, logitQnA0::Vector, gn1::Vector, A, Y;
              param::Parameter=ATE(),
              weightedfluc::Bool=false)
    q = Qmodel(logitQnA1, logitQnA0)
    fluctuate!(q, param, gn1, A, Y, weighted=weightedfluc)
    TMLE(applyparam(param, q, A, Y)..., nobs(q), estimand(param))
end

end
