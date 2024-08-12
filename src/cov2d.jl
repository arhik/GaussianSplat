
function computeCov2d_kernel(cov2ds, rots, scalesGPU)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    R = MArray{Tuple{2,2},Float32}(undef)
    theta = rots[1, idx]
    cos𝚯 = CUDA.cos(theta)
    sin𝚯 = CUDA.sin(theta)
    R[1, 1] = cos𝚯
    R[1, 2] = -sin𝚯
    R[2, 1] = sin𝚯
    R[2, 2] = cos𝚯
    S = MArray{Tuple{2,2},Float32}(undef)
    S[1, 1] = exp(scalesGPU[1, idx])
    S[1, 2] = 0.0f0
    S[2, 1] = 0.0f0
    S[2, 2] = exp(scalesGPU[2, idx])
    W = R * S
    J = W * adjoint(W)
    for i in 1:2
        for j in 1:2
            cov2ds[i, j, idx] = J[i, j]
        end
    end
    cov2ds[1, 1, idx] += 0.3
    cov2ds[2, 2, idx] += 0.3
    return
end

function computeInvCov2d(cov2ds, invCov2ds)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    R = MArray{Tuple{2,2},Float32}(undef)
    for i in 1:2
        for j in 1:2
            R[i, j] = cov2ds[i, j, idx]
        end
    end
    invCov2d = CUDA.inv(R)
    for i in 1:2
        for j in 1:2
            invCov2ds[i, j, idx] = invCov2d[i, j]
        end
    end
    return
end
