using CUDA
using CUDA: i32
using Flux
using StaticArrays
using BenchmarkTools
using Infiltrator
using PlyIO

using Images
using ImageView

struct ImageData
    imageBuffer
    transmittance
end

abstract type AbstractSplatData end
abstract type AbstractSplat2DData <: AbstractSplatData end
abstract type AbstractSplat3DData <: AbstractSplatData end

mutable struct SplatData2D <: AbstractSplat2DData
    means::AbstractArray{Float32, 2}
    scales::AbstractArray{Float32, 2}
    rotations::AbstractArray{Float32, 2}
    opacities::AbstractArray{Float32, 2}
    colors::AbstractArray{Float32, 2}
end

mutable struct SplatGrads2D <: AbstractSplat2DData
    Δmeans
    Δscales
    Δrotations
    Δopacities
    Δcolors
end

# struct SplatData3D <: AbstractSplat3DData
#     means
#     scales
#     shs
#     quaternions
#     opacities
#     features
# end

function readSplatFile(path)
	plyData = PlyIO.load_ply(path);
	vertexElement = plyData["vertex"]
	sh = cat(map((x) -> getindex(vertexElement, x), ["f_dc_0", "f_dc_1", "f_dc_2"])..., dims=2)
	scale = cat(map((x) -> getindex(vertexElement, x), ["scale_0", "scale_1", "scale_2"])..., dims=2)
	# normals = cat(map((x) -> getindex(vertexElement, x), ["nx", "ny", "nz"])..., dims=2)
	points = cat(map((x) -> getindex(vertexElement, x), ["x", "y", "z"])..., dims=2)
	quaternions = cat(map((x) -> getindex(vertexElement, x), ["rot_0", "rot_1", "rot_2", "rot_3"])..., dims=2)
	features = cat(map((x) -> getindex(vertexElement, x), ["f_rest_$i" for i in 0:44])..., dims=2)
	opacity = vertexElement["opacity"] .|> sigmoid
	splatData = SplatData3D(points, scale, sh, quaternions, opacity, features) 
	return splatData
end

# function initData2D(path)
#     return nothing
# end

# function initData3D(path)
#     return nothing
# end

@enum SplatType begin 
    SPLAT2D
    SPLAT3D
    OPTIMAL_PROJECTION_SPLAT3D
end

function initData(splatType::SplatType, nGaussians::Int64; path::Union{Nothing, String} = nothing)
    n = nGaussians
    if path === nothing
        means = CUDA.rand(Float32, 2, n);
        scales = 20.0f0.*CUDA.rand(Float32, 2, n)
        rots = Float32(pi/2.0f0)*(CUDA.rand(Float32, 1, n) .- 0.5f0);
        opacities = CUDA.rand(Float32, 1, n);
        colors = CUDA.rand(Float32, 3, n);
        splatData = SplatData2D(
            means,
            scales,
            rots,
            opacities,
            colors,
        )
        return splatData
    else
        if splatType == Splat2D
            return nothing
        elseif splatType == Splat3D
            return readSplatFile(path)
        end
    end
end


function initGrads(splatData::SplatData2D)
    Δmeans = similar(splatData.means) .|> zero;
    Δrots = similar(splatData.rotations) .|> zero;
    Δcolors = similar(splatData.colors) .|> zero;
    Δscales = similar(splatData.scales) .|> zero
    Δopacities = similar(splatData.opacities) .|> zero
    splatDataGrads = SplatGrads2D(
        Δmeans,
        Δscales,
        Δrots,
        Δopacities,
        Δcolors,
    )
    return splatDataGrads
end

function resetGrads(splatData::SplatGrads2D)
    splatData.Δmeans .= 0
    splatData.Δrotations .= 0
    splatData.Δcolors .= 0
    splatData.Δscales .= 0
    splatData.Δopacities .= 0
end

function splatDraw(cimage, transGlobal, means, bbs, invCov2ds, hitIdxs, opacities, colors)
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
    invCov2d = MArray{Tuple{2, 2,}, Float32}(undef)
    delta = MArray{Tuple{2,}, Float32}(undef)
    splatData[txIdx, tyIdx, 1] = 0.0f0
    splatData[txIdx, tyIdx, 2] = 0.0f0
    splatData[txIdx, tyIdx, 3] = 0.0f0
    splatData[txIdx, tyIdx, transIdx] = 1.0f0
    # disttmp = MVector{2, Float32}(undef)
    sync_threads()
    for hIdx in 1:size(hitIdxs, 3)
        bidx = hitIdxs[bxIdx, byIdx, hIdx]
        if bidx == 0
            continue
        end
        for ii in 1:2
            for jj in 1:2
                invCov2d[ii, jj] = invCov2ds[ii, jj, bidx]
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
            disttmp  = invCov2d*delta
            dist = disttmp[1]*delta[1] + disttmp[2]*delta[2]
            alpha = opacity*exp(-dist)
            transmittance = splatData[txIdx, tyIdx, transIdx]
            splatData[txIdx, tyIdx, 1] += (colors[1, bidx]*alpha*transmittance)
            splatData[txIdx, tyIdx, 2] += (colors[2, bidx]*alpha*transmittance)
            splatData[txIdx, tyIdx, 3] += (colors[3, bidx]*alpha*transmittance)
            splatData[txIdx, tyIdx, transIdx] *= (1.0f0 - alpha)
        end
    end
    sync_threads()
    cimage[i, j, 1] = splatData[txIdx, tyIdx, 1]
    cimage[i, j, 2] = splatData[txIdx, tyIdx, 2]
    cimage[i, j, 3] = splatData[txIdx, tyIdx, 3]
    transGlobal[i, j] = splatData[txIdx, tyIdx, transIdx]
    sync_threads()
    return
end

function splatGrads(cimage, ΔC, transGlobal, bbs, hitIdxs, invCov2ds, means, meanGrads, opacities, opacityGrads, colors, colorGrads )
    w = size(cimage, 1)
    h = size(cimage, 2)
    bxIdx = blockIdx().x
    byIdx = blockIdx().y 
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    i = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1i32)*blockDim().y + threadIdx().y
    
    S = MArray{Tuple{3,}, Float32}(undef)
    Δα = 0.0f0
    αGrads = 0.0f0
    for i in 1:3
        S[i] = 0.0f0
    end
    transData = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y))
    transData[txIdx, tyIdx] = transGlobal[i, j]
    sync_threads()
    invCov2d = MArray{Tuple{2, 2,}, Float32}(undef)
    delta = MArray{Tuple{2,}, Float32}(undef)
    ΔMean = MArray{Tuple{2,}, Float32}(undef)
    ΔΣ = MArray{Tuple{2, 2}, Float32}(undef)
    rTemp = MArray{Tuple{2, 2}, Float32}(undef)
    rTemp[1]  = 0
    rTemp[2] = -1
    rTemp[3] = 1
    rTemp[4] = 0 #[0 -1; 1 0]
    Δo = 0
    Δσ = 0
    sync_threads()
    for hIdx in size(hitIdxs, 3):-1:1
        bidx = hitIdxs[bxIdx, byIdx, hIdx]
        if bidx == 0
            continue
        end
        for ii in 1:2
            for jj in 1:2
                invCov2d[ii, jj] = invCov2ds[ii, jj, bidx]
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
            disttmp  = invCov2d*delta
            dist = disttmp[1]*delta[1] + disttmp[2]*delta[2]
            ΔMean = invCov2d*delta
            ΔΣ = ΔMean*adjoint(ΔMean)
            Δo = exp(-dist)
            Δσ = -opacity*Δo
            transmittance = transData[txIdx, tyIdx]
            alpha = opacity*exp(-dist)
            Δc = alpha*transmittance
            CUDA.@atomic colorGrads[1, bidx] += (Δc*ΔC[txIdx, tyIdx, 1])
            CUDA.@atomic colorGrads[2, bidx] += (Δc*ΔC[txIdx, tyIdx, 2])
            CUDA.@atomic colorGrads[3, bidx] += (Δc*ΔC[txIdx, tyIdx, 3])
            # TODO unroll macro
            # @unroll for i in 1:3
            #     Δα[i] = (colors[i, bidx] - (S[i]/(1.0f0 -alpha)))
            # end
            Δα += (colors[1, bidx]*transmittance - (S[1]/(1.0f0 - alpha)))/3
            Δα += (colors[2, bidx]*transmittance - (S[2]/(1.0f0 - alpha)))/3
            Δα += (colors[3, bidx]*transmittance - (S[3]/(1.0f0 - alpha)))/3
            # TODO set compact gradients
            αGrads += (Δα*ΔC[txIdx, tyIdx, 1])
            αGrads += (Δα*ΔC[txIdx, tyIdx, 2])
            αGrads += (Δα*ΔC[txIdx, tyIdx, 3])
            CUDA.@atomic opacityGrads[bidx] += αGrads*Δo
            Δσ += αGrads*Δσ
            CUDA.@atomic meanGrads[1, bidx] += ΔMean[1]*Δσ
            CUDA.@atomic meanGrads[2, bidx] += ΔMean[2]*Δσ
            # ΣGrad = ΔΣ*Δσ
            # wGrad = 2.0f0*ΣGrad*W # + adjoint(ΣGrad)*W
			# rGrad = wGrad*adjoint(scale)
			# RGrad = rTemp*rot*rGrad
			# θGrad = atan(RGrad[2, 1], RGrad[2, 2])
			# sGrad = adjoint(rot)*wGrad
            # update S after updating gradients
            for i in 1:3
                S[i] += alpha*transmittance
            end
            transData[txIdx, tyIdx] *= 1.0f0/(1.0f0 - alpha)
        end
    end
    sync_threads()
    transGlobal[i, j] = transData[txIdx, tyIdx]
    sync_threads()
    return
end
