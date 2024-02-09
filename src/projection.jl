
function computeCov3dProjection_kernel(cov2ds, cov3ds, rotation, affineTransform)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    quat = quaternions[1, idx]
    @inline R = quatToRot(quat)
    S = MArray{Tuple{3, 3}, Float32}(undef)
    for i in 1:3
        for j in 1:3
            S[i, j] = 0.0f0
        end
    end
    S[1, 1] = scales[1, idx]
    S[2, 2] = scales[2, idx]
    S[3, 3] = scales[3, idx]
    W = R*S
    J = W*adjoint(W)
    for i in 1:3
        for j in 1:3
            cov3ds[i, j, idx] = J[i, j]
        end
    end
    return
end
