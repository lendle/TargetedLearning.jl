module Parameters

export Regimen,
       StaticRegimen,
       DynamicRegimen,
       regimen,

       Mean,
       ATE,

       estimand,
       applyparam,
       fluccovar

using ..Qmodels
import ..Common: Parameter, fluccovar
import StatsBase.predict

abstract Regimen{T<:FloatingPoint}

immutable StaticRegimen{T<:FloatingPoint} <: Regimen{T}
    a::T
end

# Base.getindex(d::StaticRegimen, i::Real) = d.a
# Base.getindex(d::StaticRegimen, i::AbstractVector{Bool}) = fill(d.a, sum(i))
# Base.getindex{T<:Real}(d::StaticRegimen, i::AbstractVector{T}) = fill(d.a, length(i))

immutable DynamicRegimen{T<:FloatingPoint} <: Regimen{T}
    a::Vector{T}
end

# Base.getindex(d::DynamicRegimen, idx...) = getindex(d.a, idx...)

regimen(a::Regimen) = a
regimen(a::Real) = StaticRegimen(float(a))
regimen{T<:Real}(a::Vector{T}) = DynamicRegimen(float(a))

predict{T<:FloatingPoint}(q::Qmodel{T}, d::StaticRegimen{T}) = predict(q, fill(d.a, nobs(q)))
predict{T<:FloatingPoint}(q::Qmodel{T}, d::DynamicRegimen{T}) = predict(q, d.a, nobs(q))

type Mean{T<:FloatingPoint} <: Parameter{T}
    d::Regimen{T}
end

Mean(d) = Mean(regimen(d))

type ATE{T<:FloatingPoint} <: Parameter{T}
    d1::Regimen{T}
    d0::Regimen{T}
end

ATE(d1, d0) = ATE(regimen(d1), regimen(d0))
ATE() = ATE(1.0, 0.0)

function fluccovar{T<:FloatingPoint}(param::Mean{T}, a::Vector{T}, gn1::Vector{T})
    ifelse(param.d.a .== a, 1 ./ifelse(a .== 1, gn1, 1-gn1), zero(T))
end

function fluccovar{T<:FloatingPoint}(param::ATE{T}, a::Vector{T}, gn1::Vector{T})
    invgna = 1./ifelse(a .== 1, gn1, 1-gn1)
    ifelse(param.d1.a .== a, invgna, zero(T)) .-  ifelse(param.d0.a .== a, invgna, zero(T))
end

function applyparam{T<:FloatingPoint}(param::Mean{T}, q::Qmodel{T}, gn1::Vector{T}, A, Y)
    QnAd = predict(q, param.d)
    psi = mean(QnAd)
    h = fluccovar(param, A, gn1)
    ic = h .* (Y .- QnAd) .+ QnAd .- psi
    (psi, ic)
end

function applyparam{T<:FloatingPoint}(param::ATE{T}, q::Qmodel{T}, gn1::Vector{T}, A, Y)
    QnAd1 = predict(q, param.d1)
    QnAd0 = predict(q, param.d0)
    QnAA = predict(q, A)
    Qndiff = QnAd1 .- QnAd0
    psi = mean(Qndiff)
    h = fluccovar(param, A, gn1)
    ic = h .* (Y .- QnAA) .+ Qndiff .- psi
    (psi, ic)
end

estimand(param::Parameter) = "ψ"
estimand(param::Mean) = "E(Y(d))"
estimand(param::ATE) = "ATE"

end
