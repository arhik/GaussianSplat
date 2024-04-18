# Compute Bounding Boxes 
function computeBB(cov2ds, bbs, means, sz)
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
    sync_threads()
    Δ = CUDA.det(cov2d)
    if Δ == 0.0f0
        return
    end
    halfad = (cov2d[1] + cov2d[4])/2.0f0
    eigendir1 = halfad - sqrt(max(0.1, halfad*halfad - Δ))
    eigendir2 = halfad + sqrt(max(0.1, halfad*halfad - Δ))
    r = ceil(3.0*sqrt(max(eigendir1, eigendir2)))
    BB[1, 1] = max(1, r*BB[1, 1] + means[1, idx] |> floor)
    BB[1, 2] = min(sz[1], r*BB[1, 2] + means[1, idx] |> ceil)
    BB[2, 1] = max(1, r*BB[2, 1] + means[2, idx] |> floor)
    BB[2, 2] = min(sz[2], r*BB[2, 2] + means[2, idx] |> ceil)
    sync_threads()
    for i in 1:2
        for j in 1:2
            bbs[i, j, idx] = BB[i, j]
        end
    end
    sync_threads()
    return
end
