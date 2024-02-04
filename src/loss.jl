
using Images
using Flux
using Statistics

function kernelWindow(windowSize, σ)
    kernel = zeros(windowSize, windowSize)
    dist = C -> sqrt(sum((ceil.(windowSize./2) .- C.I).^2))
    for idx in CartesianIndices((1:windowSize, 1:windowSize))
        kernel[idx] = exp(-(dist(idx)))/sqrt(2*σ^2)
    end
    return (kernel/sum(kernel))
end

using TestImages

using ImageQualityIndexes

windowSize = 11

function initKernel(x, y, windowSize)
    nChannels = size(x, 3)
    kernel = repeat(
        reshape(kernelWindow(windowSize, 1.5) .|> Float32, (windowSize, windowSize, 1, 1)),
            inner=(1, 1, 1, nChannels)
        ) |> gpu
    cdims = DenseConvDims(
        size(x), 
        size(kernel), 
        stride=(1, 1), 
        padding=div(windowSize, 2), 
        dilation=(1, 1), 
        flipkernel=false, 
        groups=nChannels          
    )
    return (kernel, cdims)
end

C1 = 0.01f0^2
C2 = 0.03f0^2

function ssimScore(x, y)
    μx = conv(x, kernel, cdims)
    μy = conv(y, kernel, cdims)

    μx2 = μx.^2
    μy2 = μy.^2
    μxy = μx.*μy

    σ2x = conv(x.^2, kernel, cdims) .- μx2
    σ2y = conv(y.^2, kernel, cdims) .- μy2
    σxy = conv(x.*y, kernel, cdims) .- μxy

    lp = (2.0f0.*μxy .+ C1)./(μx2 .+ μy2 .+ C1)
    cp = (2.0f0.*σxy .+ C2)./(σ2x .+ σ2y .+ C2)

    ssimMap = lp.*cp
    return mean(ssimMap)
end

ssimLoss(x, y) = -ssimScore(x, y)

dssim(x, y) = 1.0f0 - ssimScore(x, y)

function loss(img, gt)
	λ = 0.1
	return (1-λ)*sum(abs.(img .- gt))/(2.0*length(img)) + λ*dssim(img, gt)/2.0
end
