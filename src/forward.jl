
# Binning
# TODO powers of 2
function preprocess(renderer::GaussianRenderer2D)
    means = renderer.splatData.means
    cov2ds = renderer.cov2ds;
    bbs = renderer.bbs ;
    invCov2ds = renderer.invCov2ds;
    rots = renderer.splatData.rotations;
    scales = renderer.splatData.scales;
    @cuda threads=32 blocks=div(n, 32) computeCov2d_kernel(cov2ds, rots, scales)
    @cuda threads=32 blocks=div(n, 32) computeInvCov2d(cov2ds, invCov2ds)
    @cuda threads=32 blocks=div(n, 32) computeBB(cov2ds, bbs, means, size(renderer.imageData)[1:end-1])
end

function compactIdxs(renderer)
    bbs = renderer.bbs ;
    hits = CUDA.zeros(UInt8, blocks..., renderer.nGaussians);
    @cuda threads=32 blocks=div(n, 32) hitBinning(hits, bbs, threads..., blocks...)
    hitScans = CUDA.zeros(UInt16, size(hits));
    CUDA.scan!(+, hitScans, hits; dims=3);
    maxHits = maximum(hitScans) |> UInt16
    maxBinSize = min(4096, nextpow(2, maxHits)) # TODO limiting maxBinSize hardcoded to 4096
    hitIdxs = CUDA.zeros(UInt32, blocks..., maxBinSize);
    @cuda threads=blocks blocks=(32, div(n, 32)) shmem=reduce(*, blocks)*sizeof(UInt32) compactHits(
        hits, 
        bbs, 
        hitScans, 
        hitIdxs
    )
    return hitIdxs
end

function forward(renderer, hitIdxs)
    cimage = renderer.imageData
    transmittance = renderer.transmittance
    means = renderer.splatData.means
    bbs = renderer.bbs
    opacities = renderer.splatData.opacities
    colors = renderer.splatData.colors
    @cuda threads=threads blocks=blocks shmem=(4*(reduce(*, threads))*sizeof(Float32)) splatDraw(
        cimage, 
        transmittance,
        means, 
        bbs,
        hitIdxs,
        opacities,
        colors
    )
end

# TODO define config and determine threads
# splatDrawConfig = 
