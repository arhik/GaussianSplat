using CUDA
using CUDA: i32
using Flux
using StaticArrays
using BenchmarkTools

using Images
using ImageView

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

function computeInvCov2d(cov2ds, invCov2ds)
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    R = MArray{Tuple{2, 2}, Float32}(undef)
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
    Δ = CUDA.det(cov2d)

    halfad = (cov2d[1] + cov2d[4])/2.0f0
    eigendir1 = halfad - sqrt(max(0.1, halfad*halfad - Δ))
    eigendir2 = halfad + sqrt(max(0.1, halfad*halfad - Δ))
    r = ceil(3.0*sqrt(max(eigendir1, eigendir2)))
    BB[1, 1] = max(1, r*BB[1, 1] + sz[1]*means[1, idx] |> floor)
    BB[1, 2] = min(sz[1], r*BB[1, 2] + sz[1]*means[1, idx] |> ceil)
    BB[2, 1] = max(1, r*BB[2, 1] + sz[2]*means[2, idx] |> floor)
    BB[2, 2] = min(sz[2], r*BB[2, 2] + sz[2]*means[2, idx] |> ceil)
    for i in 1:2
        for j in 1:2
            bbs[i, j, idx] = BB[i, j]
        end
    end
    return
end

function hitBinning(hits, bbs, blockSizeX, blockSizeY, gridSizeX, gridSizeY)
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    xbbmin = bbs[1, 1, idx]
    xbbmax = bbs[1, 2, idx]
    ybbmin = bbs[2, 1, idx]
    ybbmax = bbs[2, 2, idx]

    bminxIdx = UInt32((div(xbbmin, float(blockSizeX)))) + 1i32
    bminyIdx = UInt32((div(ybbmin, float(blockSizeY)))) + 1i32
    bmaxxIdx = UInt32((div(xbbmax, float(blockSizeX)))) + 1i32
    bmaxyIdx = UInt32((div(ybbmax, float(blockSizeY)))) + 1i32

    # BB Cover 
    for i in bminxIdx:bmaxxIdx
        for j in bminyIdx:bmaxyIdx
            if i <= gridSizeX && j <= gridSizeY
                hits[i, j, idx] = 1
            end
        end
    end
    
    return
end

function compactHits(hits, bbs, hitscan, hitIdxs)
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    bxIdx = blockIdx().x
    byIdx = blockIdx().y
    bIdx = gridDim().x*(byIdx - 1i32) + bxIdx
    shmem = CuDynamicSharedArray(UInt32, (blockDim().x, blockDim().y))
    shmem[txIdx, tyIdx] = hitscan[txIdx, tyIdx, bIdx]
    sync_threads()
    if hits[txIdx, tyIdx, bIdx] == 1
        idx = shmem[txIdx, tyIdx]
        hitIdxs[txIdx, tyIdx, idx] = bIdx
    end
    sync_threads()
    return
end


function splatDraw(cimage, transGlobal, means, bbs, hitIdxs, opacities, colors)
    w = size(cimage, 1)
    h = size(cimage, 2)
    bxIdx = blockIdx().x
    byIdx = blockIdx().y 
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    i = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1i32)*blockDim().y + threadIdx().y
    
    transIdx = 4
    splatData = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y, 4))
    splatData[txIdx, tyIdx, 1] = 0.0f0
    splatData[txIdx, tyIdx, 2] = 0.0f0
    splatData[txIdx, tyIdx, 3] = 0.0f0
    splatData[txIdx, tyIdx, transIdx] = 1.0f0
    sync_threads()

    for hIdx in 1:size(hitIdxs, 3)
        bidx = hitIdxs[bxIdx, byIdx, hIdx]
        if bidx == 0
            continue
        end
        xbbmin = bbs[1, 1, bidx]
        ybbmin = bbs[2, 1, bidx]
        xbbmax = bbs[1, 2, bidx]
        ybbmax = bbs[2, 2, bidx]
        hit = (xbbmin <= i <= xbbmax) && (ybbmin <= j <= ybbmax)
        opacity = opacities[bidx]
        if hit==true
            deltaX = float(i) - w*means[1, bidx]
            deltaY = float(j) - h*means[2, bidx]
            dist  = sqrt(deltaX*deltaX + deltaY*deltaY)/1.0
            transmittance = splatData[txIdx, tyIdx, transIdx]
            alpha = opacity*exp(-dist)
            splatData[txIdx, tyIdx, 1] += colors[1, bidx]*alpha*transmittance
            splatData[txIdx, tyIdx, 2] += colors[2, bidx]*alpha*transmittance
            splatData[txIdx, tyIdx, 3] += colors[3, bidx]*alpha*transmittance
            splatData[txIdx, tyIdx, transIdx] *= (1.0f0 - alpha)
        end
    end
    sync_threads()
    cimage[i, j, 1] += splatData[txIdx, tyIdx, 1]
    cimage[i, j, 2] += splatData[txIdx, tyIdx, 2]
    cimage[i, j, 3] += splatData[txIdx, tyIdx, 3]
    transGlobal[i, j] = splatData[txIdx, tyIdx, transIdx]
    return
end

function splatGrads(cimage, transGlobal, bbs, hitIdxs, invCov2ds, means, meanGrads, opacities, opacityGrads, colors, colorGrads)
    w = size(cimage, 1)
    h = size(cimage, 2)
    bxIdx = blockIdx().x
    byIdx = blockIdx().y 
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    i = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1i32)*blockDim().y + threadIdx().y
    
    S = MArray{Tuple{3,}, Float32}(undef)
    Δα = MArray{Tuple{3,}, Float32}(undef)
    for i in 1:3
        S[i] = 0.0f0
        Δα[i] = 0.0f0
    end
    transIdx = 4
    splatData = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y, 4))
    splatData[txIdx, tyIdx, 1] = 0.0f0
    splatData[txIdx, tyIdx, 2] = 0.0f0
    splatData[txIdx, tyIdx, 3] = 0.0f0
    splatData[txIdx, tyIdx, transIdx] = transGlobal[i, j]
    sync_threads()

    invCov2d = MArray{Tuple{2, 2,}, Float32}(undef)
    delta = MArray{Tuple{2,}, Float32}(undef)
    ΔMean = MArray{Tuple{2,}, Float32}(undef)
    ΔΣ = MArray{Tuple{2, 2}, Float32}(undef)
    Δo = 0
    Δσ = 0
    for hIdx in size(hitIdxs, 3):-1:1

        bidx = hitIdxs[bxIdx, byIdx, hIdx]
        if bidx == 0
            continue
        end
        for i in 1:2
            for j in 1:2
                invCov2d[i, j] = invCov2ds[i, j, bidx]
            end
        end
        xbbmin = bbs[1, 1, bidx]
        ybbmin = bbs[2, 1, bidx]
        xbbmax = bbs[1, 2, bidx]
        ybbmax = bbs[2, 2, bidx]
        hit = (xbbmin <= i <= xbbmax) && (ybbmin <= j <= ybbmax)
        opacity = opacities[bidx]
        if hit==true
            deltaX = float(i) - w*means[1, bidx]
            delta[1] = deltaX
            deltaY = float(j) - h*means[2, bidx]
            delta[2] = deltaY
            dist  = sqrt(deltaX*deltaX + deltaY*deltaY)/1.0 # TODO variance ?
            ΔMean = invCov2d*delta
            ΔΣ = ΔMean*adjoint(ΔMean)
            Δo = exp(-dist)
            Δσ = -opacity*Δo
            transmittance = splatData[txIdx, tyIdx, transIdx]
            alpha = opacity*exp(-dist)
            Δc = alpha*transmittance
            # TODO unroll macro
            # @unroll for i in 1:3
            #     Δα[i] = (colors[i, bidx] - (S[i]/(1.0f0 -alpha)))
            # end
            Δα[1] = (colors[1, bidx] - (S[1]/(1.0f0 -alpha)))
            Δα[2] = (colors[2, bidx] - (S[2]/(1.0f0 -alpha)))
            Δα[3] = (colors[3, bidx] - (S[3]/(1.0f0 -alpha)))
            # TODO set compact gradients

            # update S after updating gradients
            for i in 1:3
                S[i] += alpha*transmittance
            end
            splatData[txIdx, tyIdx, 1] += colors[1, bidx]*alpha*transmittance
            splatData[txIdx, tyIdx, 2] += colors[2, bidx]*alpha*transmittance
            splatData[txIdx, tyIdx, 3] += colors[3, bidx]*alpha*transmittance
            splatData[txIdx, tyIdx, transIdx] *= 1.0f0/(1.0f0 - alpha)
        end
    end
    transGlobal[i, j] = splatData[txIdx, tyIdx, transIdx]
    sync_threads()
    return
end

n = 32*1024

means = CUDA.rand(2, n);
Δmeans = similar(means);
rots = CUDA.rand(1, n) .- 1.0f0;
Δrots = similar(rots)
colors = CUDA.rand(3, n);
Δcolors = similar(colors)
scales = 2.0f0.*CUDA.rand(2, 2, n) .- 2.0f0;
Δscales = similar(colors)
opacities = CUDA.rand(1, n);
Δopacities = similar(opacities)
rotMats = CUDA.rand(2, 2, n);
ΔrotMats = similar(rotMats)
cov2ds = CUDA.rand(2, 2, n);
bbs = CUDA.zeros(2, 2, n);

invCov2ds = similar(cov2ds);

cimage = CUDA.zeros(Float32, 512, 512, 3);
transmittance = CUDA.ones(Float32, 512, 512)
sz = size(cimage)[1:2];
alpha = CUDA.zeros(sz);

# TODO powers of 2
@cuda threads=32 blocks=div(n, 32) computeCov2d_kernel(cov2ds, rotMats, scales)
@cuda threads=32 blocks=div(n, 32) computeInvCov2d(cov2ds, invCov2ds)
@cuda threads=32 blocks=div(n, 32) computeBB(cov2ds, bbs, means, sz)

threads = (16, 16)
blocks = (32, 32)
hits = CUDA.zeros(UInt8, blocks..., n);

@cuda threads=32 blocks=div(n, 32) hitBinning(hits, bbs, threads..., blocks...)

hitscan = CUDA.zeros(UInt16, size(hits));

CUDA.scan!(+, hitscan, hits; dims=3);

maxHits = maximum(hitscan) |> UInt16
maxBinSize = min(4096, nextpow(2, maxHits)) # TODO limiting maxBinSize hardcoded to 4096

hitIdxs = CUDA.zeros(UInt32, blocks..., maxBinSize);

@cuda threads=blocks blocks=(32, div(n, 32)) shmem=reduce(*, blocks)*sizeof(UInt32) compactHits(hits, bbs, hitscan, hitIdxs)

@cuda threads=threads blocks=blocks shmem=(4*(reduce(*, threads))*sizeof(Float32)) splatDraw(
    cimage, 
    transmittance,
    means, 
    bbs,
    hitIdxs,
    opacities,
    colors
)

@cuda threads=threads blocks=blocks shmem=(5*(reduce(*, threads))*sizeof(Float32)) splatGrads(
    cimage, 
    transmittance,
    bbs,
    hitIdxs,
    invCov2ds, 
    means,
    Δmeans,
    opacities,
    Δopacities,
    colors,
    Δcolors,
)

img = cimage |> cpu;
#img = img/maximum(img);
cimg = colorview(RGB{N0f8}, permutedims(n0f8.(img), (3, 1, 2)));

imshow(cimg)
