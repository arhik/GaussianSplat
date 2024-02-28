
# Binning
# TODO powers of 2

using LinearAlgebra
using Rotations
using CoordinateTransformations

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

function preprocess(renderer::GaussianRenderer3D)
    # Worldspace, clip space initializations
    # TODO avoid dynamic memory allocations
    ts = CUDA.zeros(4, renderer.nGaussians);
    tps = CUDA.zeros(4, renderer.nGaussians);
    μ′ = CUDA.zeros(2, renderer.nGaussians);
    (w, h) = size(renderer.imageData)[1:2];
    # Camera related params
	camerasPath = joinpath(
	    ENV["HOMEPATH"], "Downloads", "GaussianSplatting", "GaussianSplatting", "bonsai", "cameras.json"
    ) # TODO this is hardcoded
	camIdx = 2
    #camera = getCamera(camerasPath, camIdx) # defaultCamera();
    camera = defaultCamera();
    near = camera.near
    far = camera.far
    T = computeTransform(camera).linear |> gpu;
    P = computeProjection(camera, w, h).linear |> gpu;
    cx = w/2.0
    cy = w/2.0
    n = renderer.nGaussians
    fx = camera.fx |> Float32
    fy = camera.fy |> Float32
    means = renderer.splatData.means |> gpu
    cov2ds = renderer.cov2ds;
    cov3ds = renderer.cov3ds;
    bbs = renderer.bbs;
    invCov2ds = renderer.invCov2ds;
    quaternions = renderer.splatData.quaternions |> gpu;
    scales = renderer.splatData.scales |> gpu;
    n = renderer.nGaussians;

    CUDA.@sync begin @cuda threads=32 blocks=div(n, 32) frustumCulling(
            ts, tps, cov3ds, means,  μ′, fx, fy,
            quaternions, scales, T, P, w, h, cx, cy,
            cov2ds, far, near
        ) 
    end

    CUDA.@sync begin @cuda threads=32 blocks=div(n, 32) tValues(
            ts, cov3ds, fx, fy,
            quaternions, scales, cov2ds
        ) 
    end

    #renderer.positions = μ′
    renderer.camera = camera
    sortIdxs = CUDA.sortperm(tps[3, :], lt=isless) # chck
    renderer.sortIdxs = sortIdxs
    CUDA.unsafe_free!(ts)
    #CUDA.unsafe_free!(tps)
    renderer.cov2ds = cov2ds[:, :, sortIdxs]
    renderer.positions = μ′[:, sortIdxs]
    # TODO this is temporary hack
    #CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeCov2d_kernel(cov2ds, rots, scales) end
    CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeInvCov2d(renderer.cov2ds, invCov2ds) end
    CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeBB(renderer.cov2ds, bbs, renderer.positions, (w, h)) end
    return tps
end


function packedTileIds(renderer)
    bbs = renderer.bbs
    packedIds = CUDA.zeros(UInt64, nGaussians)
    CUDA.@sync begin
        @cuda threads=32 blocks=div(nGaussians, 32) binPacking(bbs, packedIds, threads..., blocks...)
    end
end


function compactIdxs(renderer)
    bbs = renderer.bbs
    hits = CUDA.zeros(UInt8, blocks..., renderer.nGaussians);
    n = renderer.nGaussians
    CUDA.@sync begin 
        @cuda threads=32 blocks=div(n, 32) hitBinning(hits, bbs, threads..., blocks...)
    end

    # This is not memory efficient but works for small list of gaussians in tile ... 
    hitScans = CUDA.zeros(UInt16, size(hits));
    CUDA.@sync CUDA.scan!(+, hitScans, hits; dims=3);
    CUDA.@sync maxHits = CUDA.maximum(hitScans) |> Int

    # TODO hardcoding UInt16 will cause issues if number of gaussians in a Tile
    # if maxHits < typemax(UInt32)
    maxBinSize = min((typemax(UInt16) |> Int), nextpow(2, maxHits))# TODO limiting maxBinSize hardcoded to 4096
    renderer.hitIdxs  = CUDA.zeros(UInt32, blocks..., maxBinSize);
    # else
        # maxBinSize = 2*nextpow(2, maxHits)
        # renderer.hitIdxs = CUDA.zeros(UInt32, blocks..., maxBinSize);
    # end

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

function forward(renderer, tps)
    cimage = renderer.imageData
    invCov2ds = renderer.invCov2ds
    sortIdxs = renderer.sortIdxs
    transmittance = renderer.transmittance
    positions = renderer.positions
    bbs = renderer.bbs
    opacities = renderer.splatData.opacities |> gpu
    opacities = opacities[sortIdxs]
    shs = renderer.splatData.shs |> gpu
    shs = shs[:, sortIdxs]
    hitIdxs = renderer.hitIdxs
    eye = renderer.camera.eye .|> Float32 |>gpu
    lookAt = renderer.camera.lookAt .|> Float32 |> gpu
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=(4*(reduce(*, threads))*sizeof(Float32)) splatDraw(
            cimage, 
            transmittance,
            positions, 
            tps,
            bbs,
            invCov2ds,
            hitIdxs,
            opacities,
            shs,
            eye,
            lookAt
        )
    end
    return nothing
end

# TODO define config and determine threads
# splatDrawConfig = 
