function pcor(x::VecOrMat, y::VecOrMat, z::Vector)
    n = length(z)
    ρxz = cor(x,z)
    ρzy = cor(z,y)
    ρxy = cor(x,y)
    for j in 1:size(y, 2)
        ρzy_j = ρzy[1, j]
        for i in 1:size(x, 2)
            ρxz_i = ρxz[i, 1]
            ρxy[i,j] = (ρxy[i,j] - ρxz_i * ρzy_j)/sqrt((1.0 - ρxz_i * ρxz_i))/sqrt(1.0 - ρzy_j * ρzy_j)
        end
    end
    ρxy
end
