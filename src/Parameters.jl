module Parameters

using Docile
@document

export Regimen,
       StaticRegimen,
       DynamicRegimen,
       regimen,

       Mean,
       ATE,

       estimand,
       applyparam,
       fluccovar,
       regimens

using ..Qmodels
import ..Common: Parameter, fluccovar
import StatsBase.predict

"""
Represents a particular single time point treatment regimen.
"""
abstract Regimen{T<:FloatingPoint}

"""
A `StaticRegimen` sets treatment to a single value for all observations. Constructed with
`StaticRegimen(a)` where `a` is a floating point 0 or 1.
"""
immutable StaticRegimen{T<:FloatingPoint} <: Regimen{T}
    a::T
    function StaticRegimen(a)
        a == 0 || a == 1 || error(ArgumentError("a should be 0 or 1"))
        new(a)
    end
end
StaticRegimen{T<:FloatingPoint}(a::T) = StaticRegimen{T}(a)
StaticRegimen(a) = StaticRegimen(float(a))

"""
A `DynamicRegimen` sets treatment for each observation to a particular value. Constructed with
`DynamicRegimen(a)` where `a` a vector of floating point is 0 or 1s.
"""
immutable DynamicRegimen{T<:FloatingPoint} <: Regimen{T}
    a::Vector{T}
    function DynamicRegimen(a)
        all(aa -> aa == 0 || aa == 1, a) || error(ArgumentError("Elements of a should be 0 or 1"))
        new(a)
    end
end
DynamicRegimen{T<:FloatingPoint}(a::Vector{T}) = DynamicRegimen{T}(a)
DynamicRegimen(a) = DynamicRegimen(float(a))

"""
Converts it's argument `a` to a `StaticRegime` if `a` is a scalar, a `DynamicRegime` if `a` is a vector,
or returns `a` if `a` is already a `Regime`.
"""
:regimen

regimen(a::Regimen) = a
regimen(a::Real) = StaticRegimen(float(a))
regimen{T<:Real}(a::Vector{T}) = DynamicRegimen(float(a))

"Returns predicted values from a `Qmodel` for a `StaticRegimen`"
predict{T<:FloatingPoint}(q::Qmodel{T}, d::StaticRegimen{T}) = predict(q, fill(d.a, nobs(q)))
"Returns predicted values from a `Qmodel` for a `DynamicRegimen`"
predict{T<:FloatingPoint}(q::Qmodel{T}, d::DynamicRegimen{T}) = predict(q, d.a, nobs(q))

"""
The `Mean` parameter is \(E_0E_0(Y\mid A=d, W)\).

Under causal assumptions, this can be interpreted as
the mean of the counterfactual outcome under regimen `d`.
"""
type Mean{T<:FloatingPoint} <: Parameter{T}
    d::Regimen{T}
end

Mean(d) = Mean(regimen(d))

"""
The `ATE` parameter is \(E_0[E_0(Y\mid A=d1, W) - E_0(Y\mid A=d0, W)]\).

Under causal assumptions, this can be interpreted as
the difference in mean counterfactual outcome under regimens `d1` and `d0`.  When `d1` is the static regimen setting treatment to 1
and `d0` is the static regimen setting treatment to 0, this is called the averate treatment effect (ATE).
"""
type ATE{T<:FloatingPoint} <: Parameter{T}
    d1::Regimen{T}
    d0::Regimen{T}
end

ATE(d1, d0) = ATE(regimen(d1), regimen(d0))
ATE() = ATE(1.0, 0.0)

function fluccovar{T<:FloatingPoint}(param::Mean{T}, a::Vector{T})
    convert(Vector{T}, a .== param.d.a)
end

function fluccovar{T<:FloatingPoint}(param::ATE{T}, a::Vector{T})
    convert(Vector{T}, (a .== param.d1.a) .- (a .== param.d0.a))
end

"""
Computes the parameter estimate and influence curve for a particular `Parmeter` given an estimate of Q and g.

** Arguments **

* `param` - parameter
* `q` - an estimated `Qmodel`
* `g` - vector representing the estimated propensity score for each observation
* `A` - observed treatment vector
* `Y` - observed outcome vector


"""
:applyparam

function applyparam{T<:FloatingPoint}(param::Mean{T}, q::Qmodel{T}, A, Y)
    QnAd = predict(q, param.d)
    psi = mean(QnAd)
    h = weightedcovar(lastfluc(q), A)
    ic = h .* (Y .- QnAd) .+ QnAd .- psi
    (psi, ic)
end

function applyparam{T<:FloatingPoint}(param::ATE{T}, q::Qmodel{T}, A, Y)
    QnAd1 = predict(q, param.d1)
    QnAd0 = predict(q, param.d0)
    QnAA = predict(q, A)
    Qndiff = QnAd1 .- QnAd0
    psi = mean(Qndiff)
    h = weightedcovar(lastfluc(q), A)
    ic = h .* (Y .- QnAA) .+ Qndiff .- psi
    (psi, ic)
end

estimand(param::Parameter) = "ψ"
estimand(param::Mean) = "E(Y(d))"
estimand(param::ATE) = "ATE"

regimens(param::Mean) = [param.d]
regimens(param::ATE) = [param.d1, param.d0]

end
