using NumericFuns
import NumericFuns.result_type

# result_type(f::Functor{1}, t1::Type) = typeof(evaluate(f, one(t1)))
# result_type(f::Functor{2}, t1::Type, t2::Type) = typeof(evaluate(f, one(t1), one(t2)))
# result_type(f::Functor{3}, t1::Type, t2::Type, t3::Type) = typeof(evaluate(f, one(t1), one(t2), one(t3)))


type Gatoh <: Functor{2} end
evaluate(::Gatoh, gn1::Float64, A::Float64) = A == 1.0 ? 1.0 / gn1 : - 1.0 / (1.0 - gn1)
result_type(::Gatoh, a, b) = Float64

type Agtoh <: Functor{2} end
evaluate(::Agtoh, A::Float64, gn1::Float64) = A / gn1 - (1.0 - A) / (1.0 - gn1)
result_type(::Agtoh, a, b) = Float64

type GethA0 <: Functor{1} end
evaluate(::GethA0, gn1) = -1.0 / (1.0 - gn1)
result_type(::GethA0, a) = Float64

type Trunc <: Functor{3} end
evaluate(::Trunc, x, mn, mx) = max(min(x,mx), mn)
result_type(::Trunc, a, b, c) = typeof(evaluate(one(a), (one(b), one(c))))

# type Getm <: Functor{5} end
# evaluate(::Getm, A, Y, QnA0, QnA1, gn1) = A == 1.0?  (Y - QnA1)/gn1       + QnA1 - QnA0 :
#                                                     -(Y - QnA0)/(1.0-gn1) + QnA1 - QnA0

type LogisticFun <: Functor{1} end
evaluate(::LogisticFun, x) = 1.0 / (1.0 + exp (-x))
result_type(::LogisticFun, a) = Float64