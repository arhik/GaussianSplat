
# Binning
# TODO powers of 2
@cuda threads=32 blocks=div(n, 32) computeCov2d_kernel(cov2ds, rotMats, scales)
@cuda threads=32 blocks=div(n, 32) computeInvCov2d(cov2ds, invCov2ds)
@cuda threads=32 blocks=div(n, 32) computeBB(cov2ds, bbs, means, sz)
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
