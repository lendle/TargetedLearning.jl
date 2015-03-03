module Parameters

export Regimen,
       StaticRegimen,
       DynamicRegimen,
       regimen,

       Parameter,
       Mean,
       ATE,

       fluccovar

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

abstract Parameter{T<:FloatingPoint}

type Mean{T<:FloatingPoint} <: Parameter{T}
    d::Regimen{T}
end

Mean(d) = Mean(regimen(d))

type ATE{T<:FloatingPoint} <: Parameter{T}
    d1::Regimen{T}
    d2::Regimen{T}
end

ATE(d1, d2) = ATE(regimen(d1), regimen(d2))
ATE() = ATE(1.0, 0.0)

function fluccovar{T<:FloatingPoint}(param::Mean{T}, a::Vector{T}, gn1::Vector{T})
    ifelse(param.d.a .== a, 1 ./ifelse(a .== 1, gn1, 1-gn1), zero(T))
end

function fluccovar{T<:FloatingPoint}(param::ATE{T}, a::Vector{T}, gn1::Vector{T})
    invgna = 1./ifelse(a .== 1, gn1, 1-gn1)
    ifelse(param.d1.a .== a, invgna, zero(T)) .-  ifelse(param.d2.a .== a, invgna, zero(T))
end
end