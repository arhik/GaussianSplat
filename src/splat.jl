using CUDA
using CUDA: i32
using StaticArrays

n = 100

means = CUDA.rand(2, n)
rots = CUDA.rand(1, n) .- 1.0f0
colors = CUDA.rand(3, n)
scales = 2.0f0.*CUDA.rand(2, 2, n) .- 2.0f0
opacities = CUDA.rand(1, n)
rotMats = CUDA.rand(2, 2, n)
cov2ds = CUDA.rand(2, 2, n)
bbs = CUDA.zeros(2, 2, n)

function computeCov2d_kernel(cov2ds, rotMats, scalesGPU)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    R = MArray{Tuple{2, 2}, Float32}(undef)
    for i in 1:2 
        for j in 1:2
            R[i, j] = rotMats[i, j, idx]
        end 
    end
    
    S = MArray{Tuple{2, 2}, Float32}(undef)
    for k in 1:2
        for l in 1:2
            S[k, l] = scalesGPU[k, l, idx]
        end
    end
    
    W = R*S
    J = W*adjoint(W)
    for i in 1:2
        for j in 1:2
            cov2ds[i, j, idx] = J[i, j]
        end
    end
    cov2ds[1, 1, idx] += 0.05
    cov2ds[2, 2, idx] += 0.05
    return
end

function computeBB(cov2ds, bbs, sz)
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    BB = MArray{Tuple{2, 2}, Float32}(undef)
    for i in 1:2
        for j in 1:2
            BB[i, j] = (j == 1 ? -1 : 1)
        end
    end
    cov2d = MArray{Tuple{2, 2}, Float32}(undef)
    for i in 1:2
        for j in 1:2
            cov2d[i, j] = cov2ds[i, j, idx]
        end
    end
    Δ = CUDA.det(cov2d)
    halfad = (cov2d[1] + cov2d[4])/2.0f0
    eigendir1 = halfad - sqrt(max(0.1, halfad*halfad - Δ))
    eigendir2 = halfad + sqrt(max(0.1, halfad*halfad - Δ))
    r = ceil(3.0*sqrt(max(eigendir1, eigendir2)))
    for i in 1:2
        for j in 1:2
            bbs[i, j, idx] = r*BB[i, j]
        end
    end
    BB[1, 1] = max(1, BB[1, 1])
    BB[1, 2] = min(BB[1, 2], sz[1])
    BB[2, 1] = max(1, BB[2, 1])
    BB[2, 2] = min(BB[2, 2], sz[2])
    return
end

cimage = CUDA.rand(Float32, 512, 512, 3)

sz = size(cimage)[1:2]
@cuda threads=100 blocks=1 computeCov2d_kernel(cov2ds, rotMats, scales)
@cuda threads=100 blocks=1 computeBB(cov2ds, bbs, sz)

