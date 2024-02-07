
# Binning
# TODO powers of 2
function preprocess(renderer::GaussianRenderer2D)
    means = renderer.splatData.means
    cov2ds = renderer.cov2ds;
    bbs = renderer.bbs ;
    invCov2ds = renderer.invCov2ds;
    rots = renderer.splatData.rotations;
    scales = renderer.splatData.scales;
    n = renderer.nGaussians
    bbs = renderer.bbs
    CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeCov2d_kernel(cov2ds, rots, scales) end
    CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeInvCov2d(cov2ds, invCov2ds) end
    CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeBB(cov2ds, bbs, means, size(renderer.imageData)[1:end-1]) end
    return nothing
end

function compactIdxs(renderer)
    bbs = renderer.bbs
    hits = CUDA.zeros(UInt8, blocks..., renderer.nGaussians);
    n = renderer.nGaussians
    CUDA.@sync begin 
        @cuda threads=32 blocks=div(n, 32) hitBinning(hits, bbs, threads..., blocks...)
    end
    hitScans = CUDA.zeros(UInt16, size(hits));
    CUDA.@sync CUDA.scan!(+, hitScans, hits; dims=3);
    CUDA.@sync maxHits = CUDA.maximum(hitScans)
    maxBinSize = min(4096, nextpow(2, maxHits)) # TODO limiting maxBinSize hardcoded to 4096
    renderer.hitIdxs  = CUDA.zeros(UInt32, blocks..., maxBinSize);
    CUDA.@sync begin
        @cuda threads=blocks blocks=(32, div(n, 32)) shmem=reduce(*, blocks)*sizeof(UInt32) compactHits(
            hits, 
            bbs, 
            hitScans, 
            renderer.hitIdxs
        )
    end
    CUDA.unsafe_free!(hits)
    CUDA.unsafe_free!(hitScans)
    return nothing
end

function forward(renderer)
    cimage = renderer.imageData
    invCov2ds = renderer.invCov2ds
    transmittance = renderer.transmittance
    means = renderer.splatData.means
    bbs = renderer.bbs
    opacities = renderer.splatData.opacities
    colors = renderer.splatData.colors
    hitIdxs = renderer.hitIdxs
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=(4*(reduce(*, threads))*sizeof(Float32)) splatDraw(
            cimage, 
            transmittance,
            means, 
            bbs,
            invCov2ds,
            hitIdxs,
            opacities,
            colors
        )
    end
    return nothing
end

# TODO define config and determine threads
# splatDrawConfig = 
