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
        scales = 10.0f0.*CUDA.rand(Float32, 2, n)
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
        bIdx = hitIdxs[bxIdx, byIdx, hIdx]
        if bIdx == 0
            continue
        end
        for ii in 1:2
            for jj in 1:2
                invCov2d[ii, jj] = invCov2ds[ii, jj, bIdx]
            end
        end
        xbbmin = bbs[1, 1, bIdx]
        ybbmin = bbs[2, 1, bIdx]
        xbbmax = bbs[1, 2, bIdx]
        ybbmax = bbs[2, 2, bIdx]
        hit = (xbbmin <= i <= xbbmax) && (ybbmin <= j <= ybbmax)
        opacity = opacities[bIdx]
        if hit==true
            deltaX = float(i) - w*means[1, bIdx]
            delta[1] = deltaX
            deltaY = float(j) - h*means[2, bIdx]
            delta[2] = deltaY
            disttmp  = invCov2d*delta
            dist = disttmp[1]*delta[1] + disttmp[2]*delta[2]
            alpha = opacity*exp(-dist)
            transmittance = splatData[txIdx, tyIdx, transIdx]
            splatData[txIdx, tyIdx, 1] += (colors[1, bIdx]*alpha*transmittance)
            splatData[txIdx, tyIdx, 2] += (colors[2, bIdx]*alpha*transmittance)
            splatData[txIdx, tyIdx, 3] += (colors[3, bIdx]*alpha*transmittance)
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

function splatGrads(
    cimage, ΔC, transGlobal, bbs, hitIdxs, invCov2ds, 
    means, meanGrads, opacities, opacityGrads, colors, colorGrads,
    rots, rotGrads, scales, scaleGrads
)
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
    RotMatrix = MArray{Tuple{2, 2}, Float32}(undef)
    ScaleMatrix = MArray{Tuple{2, 2}, Float32}(undef)
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
    c = 0.0f0
    sync_threads()
    for hIdx in size(hitIdxs, 3):-1:1
        bIdx = hitIdxs[bxIdx, byIdx, hIdx]
        if bIdx == 0
            continue
        end
        for ii in 1:2
            for jj in 1:2
                invCov2d[ii, jj] = invCov2ds[ii, jj, bIdx]
            end
        end
        theta = rots[1, bIdx]
        RotMatrix[1, 1] = CUDA.cos(theta)
        RotMatrix[1, 2] = -CUDA.sin(theta)
        RotMatrix[2, 1] = CUDA.sin(theta)
        RotMatrix[2, 2] = CUDA.cos(theta)
        ScaleMatrix = MArray{Tuple{2, 2}, Float32}(undef)
        ScaleMatrix[1, 1] = scales[1, bIdx]
        ScaleMatrix[1, 2] = 0.0f0
        ScaleMatrix[2, 1] = 0.0f0
        ScaleMatrix[2, 2] = scales[2, bIdx]
        W = RotMatrix*ScaleMatrix
        xbbmin = bbs[1, 1, bIdx]
        ybbmin = bbs[2, 1, bIdx]
        xbbmax = bbs[1, 2, bIdx]
        ybbmax = bbs[2, 2, bIdx]
        hit = (xbbmin <= i <= xbbmax) && (ybbmin <= j <= ybbmax)
        opacity = opacities[bIdx]
        if hit==true
            deltaX = float(i) - w*means[1, bIdx]
            delta[1] = deltaX
            deltaY = float(j) - h*means[2, bIdx]
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
            δc1 = Δc*ΔC[txIdx, tyIdx, 1]
            δc2 = Δc*ΔC[txIdx, tyIdx, 2]
            δc3 = Δc*ΔC[txIdx, tyIdx, 3]
            CUDA.@atomic colorGrads[1, bIdx] += (δc1)
            CUDA.@atomic colorGrads[2, bIdx] += (δc2)
            CUDA.@atomic colorGrads[3, bIdx] += (δc3)
            # TODO unroll macro
            # @unroll for i in 1:3
            #     Δα[i] = (colors[i, bIdx] - (S[i]/(1.0f0 -alpha)))
            # end
            Δα1 = (colors[1, bIdx]*transmittance - (S[1]/(1.0f0 - alpha)))/3
            Δα2 = (colors[2, bIdx]*transmittance - (S[2]/(1.0f0 - alpha)))/3
            Δα3 = (colors[3, bIdx]*transmittance - (S[3]/(1.0f0 - alpha)))/3
            # TODO set compact gradients
            αGrads += (Δα1*δc1)
            αGrads += (Δα2*δc2)
            αGrads += (Δα3*δc3)
            CUDA.@atomic opacityGrads[bIdx] += αGrads*Δo
            σGrad = αGrads*Δσ
            CUDA.@atomic meanGrads[1, bIdx] += ΔMean[1]*σGrad
            CUDA.@atomic meanGrads[2, bIdx] += ΔMean[2]*σGrad
            ΣGrad = ΔΣ*σGrad
            wGrad = ΣGrad*W + adjoint(ΣGrad)*W
			rGrad = wGrad*adjoint(ScaleMatrix)
			rTemp2 = rTemp*RotMatrix
            RGrad = rTemp2*rGrad
			rotGrad = CUDA.atan(RGrad[2, 1], RGrad[2, 2])
            CUDA.@atomic rotGrads[1, bIdx] += rotGrad
			scaleGrad = adjoint(RotMatrix)*wGrad
            CUDA.@atomic scaleGrads[1, bIdx] += scaleGrad[1]
            CUDA.@atomic scaleGrads[2, bIdx] += scaleGrad[2]
            # # update S after updating gradients
            S[1] += colors[1, bIdx]*Δc
            S[2] += colors[2, bIdx]*Δc
            S[3] += colors[3, bIdx]*Δc
            # for ii in 1:3
            #     c = colors[ii, bIdx]
            #     S[ii] += c*Δc
            # end
            transData[txIdx, tyIdx] *= (1.0f0/(1.0f0 - alpha))
        end
    end
    sync_threads()
    transGlobal[i, j] = transData[txIdx, tyIdx]
    sync_threads()
    return
end
