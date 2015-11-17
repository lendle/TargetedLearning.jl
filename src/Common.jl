module Common

VERSION < v"0.4-" && using Docile

export Parameter,
       fluccovar,

       Estimate,
       AbstractScalarEstimate,
       ScalarEstimate,

       name!,
       coef,
       vcov,
       nobs,
       stderr,
       confint,
       coeftable

using Distributions, Calculus, NumericExtensions

import Base.var
#borrowing some names from StatsBase, but not the StatsModels type
import StatsBase: coef, vcov, nobs, stderr, confint, coeftable, CoefTable

"""
Represents a parameter of interest. Subtypes of `Parameter` define a particular
parameter of the observed data distribution.
"""
abstract Parameter{T<:AbstractFloat}
"""
Computes the fluctuation covariate for a particular parameter.
"""
:fluccovar
fluccovar{T<:AbstractFloat}(param::Parameter{T}, a::Vector{T}, gn1::Vector{T}) = error("fluccovar not implemented for $T")


"""
Represents an estimate along with it's estimated influence curve.
"""
abstract Estimate

"Computes the influence curve based standard error(s) of an `Estimate`"
:stderr
"""
Computes influence curve based confidence interval(s) of an `Estimate`

** Arguments **

* `level` - defaults to 0.95
"""
:confint

"Returns the coefficient vector of an `Estimate`"
:coef
"Returns the number of observations an `Estimate` is based on"
:nobs
"Computes the influence curve based covariance matrix of an `Estimate`"
:vcov
"Returns the parameter name(s) of an `Estimate`"
:paramnames
"Returns a table with estimate values, standard errors, and 95% confidence limits"
:coeftable

stderr(est::Estimate) = sqrt(diag(vcov(est)))
function confint(est::Estimate, level=0.95)
    [coef(est) coef(est)] .+ [1. -1.] .* quantile(Normal(),(1. - level)/2.) .* stderr(est)
end

abstract AbstractScalarEstimate <: Estimate

coef(est::AbstractScalarEstimate) = [est.psi]
nobs(est::AbstractScalarEstimate) = est.n
vcov(est::AbstractScalarEstimate) = reshape([var(est.ic)/nobs(est)], 1,1)
paramnames(est::AbstractScalarEstimate) = [est.estimand]

function coeftable(est::AbstractScalarEstimate)
    CoefTable([coef(est) stderr(est) confint(est)],
              ["Estimate", "Std. Error", "Lower 95% CL", "Upper 95% CL"],
              paramnames(est))
end

type ScalarEstimate{T <: AbstractFloat} <: AbstractScalarEstimate
    psi::T
    ic::Vector{T}
    n::Int
    estimand::AbstractString
    function ScalarEstimate(psi, ic, estimand)
        meanic=mean(ic)
        #cutoff chosen because var(ic) = mean(ic.^2) - mean(ic)^2
        #which is greater than mean(ic.^2) - eps(T), and eps(T) is pretty small
        abs(meanic) <= sqrt(eps(T)) || error("Mean of ic should be approximately 0. It is $meanic.")
        subtract!(ic, meanic)
        new(psi, ic, length(ic), estimand)
    end
end

ScalarEstimate{T<:AbstractFloat}(psi::T, ic::Vector{T}, estimand="psi") = ScalarEstimate{T}(psi,ic, estimand)

"Sets the parameter name of a scalar estimate"
name!(est::AbstractScalarEstimate, estimand) = (est.estimand=estimand; est)

function Base.show(io::IO, t::AbstractScalarEstimate)
    println(io, "Estimate")
    show(io, coeftable(t))
end


+(a::AbstractScalarEstimate, b::AbstractScalarEstimate) = ScalarEstimate(a.psi + b.psi, a.ic .+ b.ic)
+(a::AbstractScalarEstimate, b::Real) = ScalarEstimate(a.psi + b, a.ic)
+(a::Real, b::AbstractScalarEstimate) = ScalarEstimate(a + b.psi, b.ic)

-(a::AbstractScalarEstimate, b::AbstractScalarEstimate) = ScalarEstimate(a.psi - b.psi, a.ic .- b.ic)
-(a::AbstractScalarEstimate, b::Real) = ScalarEstimate(a.psi - b, a.ic)
-(a::Real, b::AbstractScalarEstimate) = ScalarEstimate(a - b.psi, -b.ic)
-(a::AbstractScalarEstimate) = ScalarEstimate(a.psi, -a.ic)

*(a::AbstractScalarEstimate, b::AbstractScalarEstimate) = ScalarEstimate(a.psi * b.psi, a.ic .* b.psi .+ b.ic .* a.psi)
*(a::AbstractScalarEstimate, b::Real) = ScalarEstimate(a.psi * b, a.ic .* b)
*(a::Real, b::AbstractScalarEstimate) = ScalarEstimate(a * b.psi, a .* b.ic)

/(a::AbstractScalarEstimate, b::AbstractScalarEstimate) = ScalarEstimate(a.psi / b.psi, a.ic./b.psi .- b.ic .* (a.psi/b.psi/b.psi))
/(a::AbstractScalarEstimate, b::Real) = ScalarEstimate(a.psi / b, a.ic./b)
/(a::Real, b::AbstractScalarEstimate) = ScalarEstimate(a / b.psi, - b.ic .* (a/b.psi/b.psi))

^(a::AbstractScalarEstimate, b::AbstractScalarEstimate) = ScalarEstimate(a.psi ^ b.psi, (b.psi*(a.psi^(b.psi-1))) .* a.ic
                                                                         .+ (a.psi^b.psi * log(a.psi)) .* b.ic)
^(a::AbstractScalarEstimate, b::Integer) = ScalarEstimate(a.psi ^ b, (b*(a.psi^(b-1))) .* a.ic)
^(a::AbstractScalarEstimate, b::Real) = ScalarEstimate(a.psi ^ b, (b*(a.psi^(b-1))) .* a.ic)
^(a::AbstractScalarEstimate, b::Real) = ScalarEstimate(a.psi ^ b, (b*(a.psi^(b-1))) .* a.ic)
^(a::Real, b::AbstractScalarEstimate) = ScalarEstimate(a ^ b.psi, (a^b.psi * log(a)) .* b.ic)

for (funsym, _) in Calculus.derivative_rules
    fullfunpath = Expr(:.,:Base,Base.Meta.quot(funsym))
    @eval function $(fullfunpath)(a::AbstractScalarEstimate)
        psi  = a.psi
        phi_psi = $funsym(psi)
        dphi_dpsi = $(differentiate(:($funsym(psi)), :psi))
        ScalarEstimate(phi_psi, dphi_dpsi .* a.ic)
    end
end

end
