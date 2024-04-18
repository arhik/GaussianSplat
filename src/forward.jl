
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
    CUDA.@sync begin 
        @cuda threads=32 blocks=div(n, 32) computeCov2d_kernel(cov2ds, rots, scales) 
    end
    CUDA.@sync begin 
        @cuda threads=32 blocks=div(n, 32) computeInvCov2d(cov2ds, invCov2ds) 
    end
    CUDA.@sync begin 
        @cuda threads=32 blocks=div(n, 32) computeBB(
            cov2ds,
            bbs,
            means,
            size(renderer.imageData)[1:end-1]
        )
    end
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
	    ENV["HOMEPATH"], 
	    "Downloads", 
	    "GaussianSplatting", 
	    "GaussianSplatting", 
	    "bicycle", 
	    "cameras.json"
    ) # TODO this is hardcoded
	camIdx = 1
    # camera = getCamera(camerasPath, camIdx) # defaultCamera();
    camera = defaultCamera();
    near = camera.near
    far = camera.far
    T = computeTransform(camera).linear |> gpu;
    P = computeProjection(camera, w, h).linear |> gpu;
    cx = w/2.0
    cy = h/2.0
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
            ts, tps, μ′,  # outs
            means, T, P,  # ins
            w, h, cx, cy, # Numbers
        )
    end

    inFrustum = findall(x->x<-1.0f0, ts[3, :])
    ts = ts[:, inFrustum]
    n = size(ts, 2)
    renderer.nGaussians = n
    cov2ds = renderer.cov2ds[:, :, inFrustum]
    cov3ds = renderer.cov3ds[:, :, inFrustum]
    quaternions = quaternions[:, inFrustum]
    scales = scales[:, inFrustum]
    
    
    CUDA.@sync begin @cuda threads=32 blocks=div(n, 32) tValues(
            ts, cov3ds, fx, fy,
            quaternions, scales, cov2ds
        )
    end
    
    invCov2ds = renderer.invCov2ds[:, :, inFrustum]
    CUDA.@sync begin   
        @cuda threads=32 blocks=div(n, 32) computeInvCov2d(
            cov2ds, 
            invCov2ds
        ) 
    end
    bbs = bbs[:, :, inFrustum]
    μ′ = μ′[:, inFrustum]
    CUDA.@sync begin   
        @cuda threads=32 blocks=div(n, 32) computeBB(
            cov2ds, 
            bbs, 
            μ′, 
            (w, h)
        ) 
    end
    
    renderer.camera = camera
    renderer.cov2ds = cov2ds
    renderer.positions = μ′
    renderer.invCov2ds = invCov2ds
    renderer.bbs = bbs
    return (ts, tps)
end

"""
    compactIndex(renderer::Renderer)

This function compute compact indexes. 
"""
function compactIdxs(renderer, ts)
    bbs = renderer.bbs
    n = renderer.nGaussians
    hits = CUDA.zeros(UInt8, blocks..., renderer.nGaussians);
    CUDA.@sync begin 
        @cuda threads=32 blocks=div(n, 32) hitBinning(hits, bbs, threads..., blocks...)
    end
    idxs =  findall(x->x==UInt8(1), hits)
    sts = ts[3, map(i->i.I[3], idxs)]
    sIdxs = sortperm(sts, lt=!isless)
    return idxs[sIdxs]
end

function forward(renderer, tps, sortIdxs)
    cimage = renderer.imageData
    invCov2ds = renderer.invCov2ds
    transmittance = renderer.transmittance
    positions = renderer.positions
    bbs = renderer.bbs
    opacities = renderer.splatData.opacities |> gpu
    shs = renderer.splatData.shs |> gpu
    eye = renderer.camera.eye .|> Float32 |>gpu
    lookAt = renderer.camera.lookAt .|> Float32 |> gpu
    
    CUDA.@sync begin
        @cuda(
            threads=threads, 
            blocks=blocks, 
            shmem=(4*(reduce(*, threads))*sizeof(Float32)), 
            splatDraw(
                cimage, 
                transmittance, 
                positions, 
                tps,
                bbs,
                invCov2ds,
                opacities,
                shs,
                eye,
                lookAt,
                sortIdxs
            )
        )
    end
    return nothing
end

