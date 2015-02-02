module LReg

using NumericExtensions, NumericFuns, Devectorize

import StatsBase: predict, predict!

export AbstractLR, LR, SSLR, lreg, predict, predict!, sslreg, sparselreg

abstract AbstractLR

const eps = 1.0e-8

type LR{T<:FloatingPoint} <: AbstractLR
    beta::Vector{T}
end

# function lreg{T <: FloatingPoint}(X::Matrix{T}, Y::Vector{T}, offset = :none)
#     mu = mean(Y) * ones(size(Y, 1))
#     betahat = zeros(size(X, 2), 1)
#     gr = Inf*ones(size(betahat))
#     while maximum(abs(gr)) > 1.0e-10
#         resid = mu .- Y
#         gr = X'resid
#         S = (mu .* (1.0 .- mu))
#         S = reshape(S, size(S, 1))
#         SX = scale(S, X)
#         hess=BLAS.gemm('T','N', 1.0, X, SX)
# #         hess = X'SX
#         (q, r)  = qr(hess)
#         betahat-=r\q'gr
#         mu = offset == :none ? logistic(X * betahat) : logistic(X * betahat .+ offset)
#     end
#     LR(reshape(betahat, size(betahat, 1)))#vec(betahat)?
# end

type SqrtM1mM <: Functor{1} end
NumericExtensions.evaluate(::SqrtM1mM, m) = sqrt(m * (1.0 - m))

type Loss <: Functor{2} end
NumericExtensions.evaluate(::Loss, y, xb) =
    y == one(y)? log1pexp(-xb) :
    y == zero(y)? log1pexp(xb) :
    y * log1pexp(-xb) + (1.0-y) * log1pexp(xb)

log1pexp(x::Float64) =
    x <= 18.0? log1p(exp(x)) :
    x > 33.3? x :
    x + exp(-x)

log1pexp(x) = log1p(exp(x))

function xb!(xb, X, beta, offset)
    BLAS.gemv!('N', 1.0, X, beta, 0.0, vec(xb))
    if offset != :none
        add!(xb, offset)
    end
end

function lreg{T <: FloatingPoint}(X::DenseMatrix{T}, Y::Vector{T}, offset = :none; tol = eps)
    p = size(X, 2)
    mu = (Y .+ 0.5) ./ (2.0)
    #mu = fill(mean(Y), size(Y))
    beta = zeros(size(X, 2))
    oldbeta = fill(inf(T), size(beta))
    newbeta = zeros(size(beta))
    gr = fill(inf(T), size(beta))
    resid = similar(Y)
    S = similar(Y)
    SX = similar(X)
    hess = Array(T, p,p)
    xb = similar(Y)
    maxiter=25
    oldloss = Inf
    local iter
    #while maximum(abs(gr)) > 1.0e-10
    for iter in 0:maxiter
        map!(Subtract(), resid, mu, Y) #resid = mu .- Y
        BLAS.gemv!('T', 1.0, X, resid, 0.0, gr)  #gr = X'resid
        map!(SqrtM1mM(), S, mu) #S = sqrt((mu .* (1.0 .- mu)))
        #S = reshape(S, size(S, 1))
        #sqrt!(S)
        #copy!(SX,X)

        scale!(SX, S, X)
        BLAS.gemm!('T','N', 1.0, SX, SX, 0.0, hess)
        #BLAS.syrk!('U', 'T', 1.0, SX, 0.0, hess)
        qrf = qrfact!(hess)
        q = full(qrf[:Q])
        r = qrf[:R]
        # r = qrfact!(SX)[:R]
        # dir = r'r\gr
        dir = r\q'gr
        #dir = qrfact!(hess)\gr
        stepsize = 1.0
        map!(FMA(), newbeta, beta, -stepsize, dir)
        xb!(xb, X, newbeta, offset)
        newloss = mean(Loss(), Y, xb)
        while newloss - oldloss > tol && stepsize > 0.01
            #info("halving. oldloss: $oldloss, newloss: $newloss")
            stepsize /= 2.0
            map!(FMA(), newbeta, beta, -stepsize, dir)
            xb!(xb, X, newbeta, offset)
            newloss = mean(Loss(), Y, xb)
        end
        oldloss = newloss
        copy!(beta, newbeta)
        if maxabsdiff(beta, oldbeta) < tol
            break
        end
        copy!(oldbeta, beta)
        map!(LogisticFun(), mu, xb)
    end
    if iter == maxiter
        warn("lreg did not converge in $maxiter iterations.")
    end
    LR(beta)
end

lreg{T<:FloatingPoint}(X::Vector{T}, Y::Vector{T}, offset = :none) =
    lreg(reshape(X, size(X,1), 1), Y, offset)

# function predict(lr::AbstractLR, newX::DenseMatrix, kind::Symbol=:link; offset = :none)
#     @assert size(newX, 2) == length(lr.beta)
#     if kind == :prob
#         offset == :none ? logistic(newX * lr.beta) : logistic(newX * lr.beta .+ offset)
#     elseif kind == :link
#         offset == :none ? newX * lr.beta : newX * lr.beta .+ offset
#     else
#         error("kind should be :prob or :link")
#     end
# end

predict{T <: FloatingPoint}(lr::AbstractLR, newX::DenseMatrix{T}, kind::Symbol=:link; offset = :none) =
    predict!(lr, Array(T, size(newX, 1)), newX, kind, offset=offset)

predict(lr::AbstractLR, newX::Vector, kind::Symbol=:link; offset = :none)=
    predict(lr, reshape(newX, size(newX, 1), 1), kind, offset=offset)

function predict!{T<:FloatingPoint}(lr::AbstractLR, p::Vector{T}, newX::DenseMatrix{T}, kind::Symbol=:link; offset = :none)
#     BLAS.gemv!('N', 1.0, newX, lr.beta, 0.0, p) #p=newX*lr.beta
    A_mul_B!(p, newX, lr.beta)
    if offset != :none
        add!(p, offset)
    end

    if kind==:prob
        map1!(LogisticFun(), p)
    end
    p
end



type SSLR{T <: FloatingPoint} <: AbstractLR
    beta::Vector{T}
    idx::IntSet
end

function sslreg{T <: FloatingPoint}(X::DenseMatrix{T}, Y::Vector{T}, idx = 1:size(X, 2), offset = :none)
    idx = IntSet(idx) #convert to IntSet or make a copy
    nzp = length(idx)
    p = size(X, 2)
    mu = (Y .+ 0.5) ./ 2.0
    beta = zeros(size(X, 2))
    newbeta = zeros(size(beta))
    dir = zeros(size(beta))
    oldbeta = fill(Inf, size(beta))
    gr = fill(Inf, size(beta))
    resid = similar(Y)
    S = similar(Y)
    SX = similar(X)
    hess = Array(T, p,p)
    smallhess = Array(T, nzp, nzp)
    smallgr = fill(Inf, nzp)
    xb = similar(Y)
    maxiter=25
    oldloss = Inf
    local iter
    #while maximum(abs(gr)) > 1.0e-10
    for iter in 0:maxiter
        map!(Subtract(), resid, mu, Y) #resid = mu .- Y
        BLAS.gemv!('T', 1.0, X, resid, 0.0, gr)  #gr = X'resid
        map!(SqrtM1mM(), S, mu) #S = sqrt((mu .* (1.0 .- mu)))
        scale!(SX, S, X)
        #BLAS.syrk!('U', 'T', 1.0, SX, 0.0, hess)
        BLAS.gemm!('T','N', 1.0, SX, SX, 0.0, hess)

        #This is the different part
        for (i, idxi) in enumerate(idx)
            smallgr[i] = gr[idxi]
        end
        for (j, idxj) in enumerate(idx)
            for (i, idxi) in enumerate(idx)
                #if i <= j #if using cholfact, only need upper triangle
                    smallhess[i,j] = hess[idxi, idxj]
                #end
            end
        end
        qrf = qrfact!(smallhess)
        q = full(qrf[:Q])
        r = qrf[:R]
        smalldir = r\q'smallgr
        #smalldir = qrfact!(smallhess) \ smallgr
        dir[collect(idx)] = smalldir
        #This is the end of the different part

        stepsize = 1.0
        map!(FMA(), newbeta, beta, -stepsize, dir)
        xb!(xb, X, newbeta, offset)
        newloss = mean(Loss(), Y, xb)
        while newloss - oldloss > eps && stepsize > 0.01
            stepsize /= 2.0
            map!(FMA(), newbeta, beta, -stepsize, dir)
            xb!(xb, X, newbeta, offset)
            newloss = mean(Loss(), Y, xb)
        end
        oldloss = newloss
        copy!(beta, newbeta)
        if maxabsdiff(beta, oldbeta) < eps
            break
        end
        copy!(oldbeta, beta)
        map!(LogisticFun(), mu, xb)
    end
    if iter == maxiter
        warn("sslregid not converge in $maxiter iterations.")
        info("$idx")
    end
    SSLR(beta, idx)
end

function sparselreg{T <: FloatingPoint}(X::DenseMatrix{T}, Y::Vector{T}, idx = 1:size(X, 2), offset = :none)
    idx = IntSet(idx)
    if idx == IntSet(1:size(X,2))
        return SSLR(lreg(X,Y, offset).beta, idx)
    else
        Xsmall = X[:, collect(idx)]
        lr = lreg(Xsmall, Y, offset)
        beta = zeros(T, size(X, 2))
        beta[collect(idx)] = lr.beta
        return SSLR(beta, idx)
    end
end

function lreg_bfgs{T}(X::DenseMatrix{T}, Y::Vector{T}, offset = :none; weights=:none, tol=1.0e-8, maxiter=25)
    #http://research.microsoft.com/en-us/um/people/minka/papers/logreg/minka-logreg.pdf
    #Section 6
    n,p = size(X)
    mu = fill(0.5, n)
    resid = similar(Y)
    A = similar(Y)
    H_inv = eye(T, p)
    w = zeros(T, p)
    w_old = copy(w)
    Δw = copy(w)
    g = copy(w)
    g_old = copy(g)
    Δg = copy(w)
    xw = similar(Y)
    Xu = similar(Y)
    local iter
    for iter in 0:maxiter
        map!(Subtract(), resid, mu, Y) #resid = mu .- Y
        At_mul_B!(g, X, resid) #g = X'resid
        @devec A[:]  = mu .* (1.0 .- mu)

        if iter > 0
            #can't update H_inv on the first iteration...
            map!(Subtract(), Δg, g, g_old)

            ΔwᵀΔg = Δw ⋅ Δg
            b = 1.0 + vec(Δg'H_inv)⋅Δg / ΔwᵀΔg
            H_invΔg = H_inv * Δg

            BLAS.syr!('U', b/ΔwᵀΔg, Δw, H_inv)
            BLAS.syr2k!('U', 'N', -1.0/ΔwᵀΔg, Δw, H_invΔg, 1.0, H_inv)
            #TODO: use a Symmetric matrix for H_inv, and replace multiplication by H_inv
            #..... below by BLAS.symm! instead of manually symmetrizing
            for i in 2:p, j in 1:i-1
                @inbounds H_inv[i,j] = H_inv[j, i]
            end
#             H_inv += (b .* Δw*Δw' .- Δw*Δg'H_inv .- H_inv*Δg*Δw') ./ ΔwᵀΔg
        end
        u = - H_inv * g
        gᵀu = g ⋅ u
        A_mul_B!(Xu, X, u)
        @devec uᵀHu = sum(A .* Xu .* Xu)

        Δw = - (gᵀu/uᵀHu) .* u

        w .+= Δw

        #TODO: Implement backtracking line search
        #TODO: https://en.wikipedia.org/wiki/Armijo_condition
        #TODO: https://en.wikipedia.org/wiki/Wolfe_conditions


        if maxabsdiff(w, w_old) < tol
            break
        end
        xb!(xw, X, w, offset)
        map!(LogisticFun(), mu, xw)
        copy!(w_old, w)
        copy!(g_old, g)
    end
    if iter == maxiter
        warn("lreg_bfgs did not converge in $maxiter iterations.")
    end
    LReg.LR(w)
end

end
