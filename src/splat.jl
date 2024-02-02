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

function hitBinning(hits, bbs, blockSizeX, blockSizeY)
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    xbbmin = bbs[1, 1, idx]
    xbbmax = bbs[1, 2, idx]
    ybbmin = bbs[2, 1, idx]
    ybbmax = bbs[2, 2, idx]
    bxIdx = UInt32(floor(div(xbbmin, float(blockSizeX))))
    byIdx = UInt32(floor(div(ybbmin, float(blockSizeY))))
    hits[bxIdx, byIdx, idx] = 1
    bxIdx = UInt32(floor(div(xbbmax, float(blockSizeX))))
    byIdx = UInt32(floor(div(ybbmax, float(blockSizeY))))
    hits[bxIdx, byIdx, idx] = 1
    return
end

function linearScan(hits, hitscan)
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    
end

function compactHits(hits, bbs, hitscan, hitIdxs)
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    bxIdx = blockIdx().x - 1i32
    byIdx = blockIdx().y - 1i32
    bIdx = gridDim().x*(byIdx) + bxIdx + 1i32
    shmem = CuDynamicSharedArray(UInt32, (blockDim().x, blockDim().y))
    shmem[txIdx, tyIdx] = hitscan[txIdx, tyIdx, bIdx]
    if hits[txIdx, tyIdx, bIdx] == 1
        idx = shmem[txIdx, tyIdx]
        hitIdxs[txIdx, tyIdx, idx] = bIdx
    end
    return
end

function splatDraw(cimage, means, bbs, hitIdxs, opacities, colors)
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

    # compactHitIdxs = CuDynamicSharedArray(UInt32, 4096, reduce(*, size(splatData))*sizeof(Float32))
    # for chIdx in 1:size(hitIdxs, 3)
    #     compactHitIdxs[chIdx] = hitIdxs[bxIdx, byIdx, chIdx]
    # end

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
    cimage[i, j, 1] += splatData[txIdx, tyIdx, 1]
    cimage[i, j, 2] += splatData[txIdx, tyIdx, 2]
    cimage[i, j, 3] += splatData[txIdx, tyIdx, 3]
    return
end


n = 1000000

means = CUDA.rand(2, n);
rots = CUDA.rand(1, n) .- 1.0f0;
colors = CUDA.rand(3, n);
scales = 2.0f0.*CUDA.rand(2, 2, n) .- 2.0f0;
opacities = CUDA.rand(1, n);
rotMats = CUDA.rand(2, 2, n);
cov2ds = CUDA.rand(2, 2, n);
bbs = CUDA.zeros(2, 2, n);

cimage = CUDA.zeros(Float32, 512, 512, 3);
sz = size(cimage)[1:2];
alpha = CUDA.zeros(sz);

# TODO powers of 2
@cuda threads=100 blocks=div(n, 100) computeCov2d_kernel(cov2ds, rotMats, scales)
@cuda threads=100 blocks=div(n, 100) computeBB(cov2ds, bbs, means, sz)

threads = (16, 16)
blocks = (32, 32)
hits = CUDA.zeros(UInt8, blocks..., n);

@cuda threads=100 blocks=div(n, 100) hitBinning(hits, bbs, threads...)

hitscan = CUDA.zeros(UInt16, size(hits));
CUDA.scan!(+, hitscan, hits; dims=3);

maxHits = maximum(hitscan) |> UInt16
maxBinSize = min(4096, nextpow(2, maxHits)) # TODO limiting maxBinSize hardcoded to 4096

hitIdxs = CUDA.zeros(UInt32, blocks..., maxBinSize);

@cuda threads=blocks blocks=(100, div(n, 100)) shmem=reduce(*, blocks)*sizeof(UInt32) compactHits(hits, bbs, hitscan, hitIdxs)

@cuda threads=threads blocks=blocks shmem=(4*(reduce(*, threads))*sizeof(Float32))   splatDraw(
    cimage, 
    means, 
    bbs,
    hitIdxs,
    opacities,
    colors
)

img = cimage |> cpu;
#img = img/maximum(img);
cimg = colorview(RGB{N0f8}, permutedims(n0f8.(img), (3, 1, 2)));

imshow(cimg)

