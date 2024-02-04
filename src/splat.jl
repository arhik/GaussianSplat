using CUDA
using CUDA: i32
using Flux
using StaticArrays
using BenchmarkTools

using Images
using ImageView


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

function splatGrads(cimage, transGlobal, bbs, hitIdxs, invCov2ds, means, meanGrads, opacities, opacityGrads, colors, colorGrads )
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
    transData = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y))
    transData[txIdx, tyIdx] = transGlobal[i, j]
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
            CUDA.@atomic meanGrads[1, bidx] += ΔMean[1]
            CUDA.@atomic meanGrads[2, bidx] += ΔMean[2]
            ΔΣ = ΔMean*adjoint(ΔMean)
            Δo = exp(-dist)
            CUDA.@atomic opacityGrads[bidx] += Δo
            Δσ = -opacity*Δo
            transmittance = transData[txIdx, tyIdx]
            alpha = opacity*exp(-dist)
            Δc = alpha*transmittance
            CUDA.@atomic colorGrads[1, bidx] += Δc
            CUDA.@atomic colorGrads[2, bidx] += Δc
            CUDA.@atomic colorGrads[3, bidx] += Δc
            # TODO unroll macro
            # @unroll for i in 1:3
            #     Δα[i] = (colors[i, bidx] - (S[i]/(1.0f0 -alpha)))
            # end
            Δα[1] = (colors[1, bidx]*transmittance - (S[1]/(1.0f0 -alpha)))
            Δα[2] = (colors[2, bidx]*transmittance - (S[2]/(1.0f0 -alpha)))
            Δα[3] = (colors[3, bidx]*transmittance - (S[3]/(1.0f0 -alpha)))
            # TODO set compact gradients
            
            # update S after updating gradients
            for i in 1:3
                S[i] += alpha*transmittance
            end
            transData[txIdx, tyIdx] *= 1.0f0/(1.0f0 - alpha)
        end
    end
    transGlobal[i, j] = transData[txIdx, tyIdx]
    sync_threads()
    return
end
