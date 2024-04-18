@inline function quatToRot(q::MVector{4, Float32})::MArray{Tuple{3, 3}, Float32}
    R = MArray{Tuple{3, 3}, Float32}(undef)
    (w, x, y, z) = q
    R[1, 1] = 1.0f0 - 2.0f0*(y*y + z*z)
    R[2, 1] = 2.0f0*(x*y + w*z)
    R[3, 1] = 2.0f0*(x*z - w*y)
    R[1, 2] = 2.0f0*(x*y - w*z)
    R[2, 2] = 1.0f0 - 2.0f0*(x*x - z*z)
    R[3, 2] = 2.0f0*(y*z + w*x)
    R[1, 3] = 2.0f0*(x*z + w*y)
    R[2, 3] = 2.0f0*(y*z - w*x)
    R[3, 3] = 1.0f0 - 2.0f0*(x*x + y*y)
    return R
end
 
function computeCov3dProjection_kernel(cov2ds, cov3ds, rotation, affineTransform)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    quat = quaternions[1, idx]
    R = quatToRot(quat)
    S = MArray{Tuple{3, 3}, Float32}(undef)
    for i in 1:3
        for j in 1:3
            S[i, j] = 0.0f0
        end
    end
    S[1, 1] = exp(scales[1, idx])
    S[2, 2] = exp(scales[2, idx])
    S[3, 3] = exp(scales[3, idx])
    W = R*S
    J = W*adjoint(W)
    for i in 1:3
        for j in 1:3
            cov3ds[i, j, idx] = J[i, j]
        end
    end
    return
end

function frustumCulling(
    ts, tps, μ′,
    meansList, T, P, 
    w, h, cx, cy,
)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    
    meanVec = MVector{4, Float32}(undef)
    meanVec[1] = meansList[1, idx]
    meanVec[2] = meansList[2, idx]
    meanVec[3] = meansList[3, idx]
    meanVec[4] = 1

    Tcw = MArray{Tuple{4, 4}, Float32}(undef)
    for ii in 1:4
        for jj in 1:4
            Tcw[ii, jj] = T[ii, jj]
        end
    end

    tstmp = Tcw*meanVec
    #tsw = tstmp[4] + 0.00000001f0
    ts[1, idx] = tstmp[1]#/tsw
    ts[2, idx] = tstmp[2]#/tsw
    ts[3, idx] = tstmp[3]#/tsw
    ts[4, idx] = tstmp[4]#/tsw

    Ptmp = MArray{Tuple{4, 4}, Float32}(undef)
    for ii in 1:4
        for jj in 1:4
            Ptmp[ii, jj] = P[ii, jj]
        end
    end

    tx = tstmp[1]
    ty = tstmp[2]
    tz = tstmp[3]
    tw = tstmp[4]

    tpstmp = Ptmp*tstmp
    tpsw = tpstmp[4] + 0.0000001f0
    tps[1, idx] = tpstmp[1]/tpsw
    tps[2, idx] = tpstmp[2]/tpsw
    tps[3, idx] = tpstmp[3]/tpsw
    tps[4, idx] = 1.0f0#tpstmp[4]/tpsw
    
    tx′ = tpstmp[1]
    ty′ = tpstmp[2]
    tz′ = tpstmp[3]
    tw′ = tpstmp[4]

    μx = ((w*tx′/tw′) + 1)/2 + cx
    μy = ((h*ty′/tw′) + 1)/2 + cy

    #if (-w < x < w) && (-h < y < h)# && (near < tz′ < far)
        μ′[1, idx] = μx
        μ′[2, idx] = μy
    #else
        # TODO zero values are used for culling checks for now
    #   μ′[1, idx] = 0.0f0
    #   μ′[1, idx] = 0.0f0
    #end
    return nothing
end


function tValues(
    ts, cov3ds, fx, fy, quaternions, scales, cov2ds
)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    tx = ts[1, idx]
    ty = ts[2, idx]
    tz = ts[3, idx]
    tw = ts[4, idx]
    
    J = MArray{Tuple{2, 3}, Float32}(undef)
    J[1] = fx/tz
    J[2] = 0
    J[3] = 0
    J[4] = fy/tz
    J[5] = -fx*tx/(tz*tz)
    J[6] = -fy*ty/(tz*tz)
                                
    quat = MVector{4, Float32}(undef)
    quat[1] = quaternions[1, idx]
    quat[2] = quaternions[2, idx]
    quat[3] = quaternions[3, idx]
    quat[4] = quaternions[4, idx]

    R = quatToRot(quat)
    S = MArray{Tuple{3, 3}, Float32}(undef)
    for i in 1:3
        for j in 1:3
            S[i, j] = 0.0f0
        end
    end
    S[1, 1] = exp(scales[1, idx]) 
    S[2, 2] = exp(scales[2, idx])
    S[3, 3] = exp(scales[3, idx])
    W = R*S
    cov3d = W*adjoint(W)
    for i in 1:3
        for j in 1:3
            cov3ds[i, j, idx] = cov3d[i, j]
        end
    end

    JR = J*R
    JCR = JR*cov3d
    cov2d = JCR*adjoint(JR)

    for ii in 1:2
        for jj in 1:2
            cov2ds[ii, jj, idx] = cov2d[ii, jj] + 0.3
        end
    end

    return nothing
end


# CUDA.@sync begin @cuda threads=32 blocks=div(n, 32) tValues(
#         ts, tps, cov3ds, meansList,  μ′, fx, fy,
#         quaternions, scales, T, P, w, h, cx, cy,
#         cov2ds,
#     ) 
# end

