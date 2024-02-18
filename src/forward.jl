
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
    
    # Camera related params
	# camerasPath = joinpath(pkgdir(WGPUgfx), "assets", "bonsai", "cameras.json")
	# camIdx = 1
    # near = 0.1f0
    # far = 100.0f0
    # camera = getCamera(camerasPath, camIdx)
    camera = defaultCamera();
    near = camera.near
    far = camera.far
    T = computeTransform(camera).linear |> MArray |> gpu;
    (w, h) = size(renderer.imageData)[1:2];
    # P = computeProjection(camera, near, far).linear |> gpu;
    P = computeProjection(camera, w, h).linear |> gpu;
    # w = camera.width
    # h = camera.height
    cx = div(w, 2)
    cy = div(h, 2)
    n = renderer.nGaussians
    fx = camera.fx
    fy = camera.fy
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

    renderer.positions = μ′
    sortIdxs = CUDA.sortperm(tps[3, :])
    CUDA.unsafe_free!(ts)
    CUDA.unsafe_free!(tps)
    renderer.cov2ds = cov2ds[:, :, sortIdxs]
    renderer.positions = μ′[:, sortIdxs]
    # TODO this is temporary hack
    #CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeCov2d_kernel(cov2ds, rots, scales) end
    CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeInvCov2d(cov2ds, invCov2ds) end
    CUDA.@sync begin   @cuda threads=32 blocks=div(n, 32) computeBB(cov2ds, bbs, renderer.positions, (w, h)) end
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

function forward(renderer)
    cimage = renderer.imageData
    invCov2ds = renderer.invCov2ds
    transmittance = renderer.transmittance
    positions = renderer.positions
    bbs = renderer.bbs
    opacities = renderer.splatData.opacities |> gpu
    colors = renderer.splatData.shs .|> sigmoid |> gpu
    hitIdxs = renderer.hitIdxs
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=(4*(reduce(*, threads))*sizeof(Float32)) splatDraw(
            cimage, 
            transmittance,
            positions, 
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
