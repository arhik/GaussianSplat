@inline function quatToRot(q::MVector{4, Float32})
    rotMat3D = MArray{Tuple{3, 3}, Float32}(undef)
    x = q[1]
    y = q[2]
    z = q[3]
    w = q[4]
    R[1] = 1.0f0 - 2.0f0*(y*y + z*z)
    R[2] = 2.0f0*(x*y - w*z)
    R[3] = 2.0f0*(x*z - w*y)
    R[4] = 2.0f0*(x*y + w*z)
    R[5] = 1.0f0 - 2.0f0*(x*x - z*z)
    R[6] = 2.0f0*(y*z - w*z)
    R[7] = 2.0f0*(x*z - w*y)
    R[8] = 2.0f0*(y*z + w*x)
    R[9] = 1.0f0 - 2.0f0*(x*x - y*y)
    return R
end

function computeCov3d_kernel(cov3ds, quaternions, scales)
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
