using TestImages
using ImageQualityIndexes
using ImageView
gtimg = testimage("coffee")

include("loss.jl")

function sizeSelection(imsize)
    pows = [(nextpow, nextpow), (nextpow, prevpow), (prevpow, nextpow), (prevpow, prevpow)]
    sizes = map(pow -> (pow[1](2, imsize[1]), pow[2](2, imsize[2])), pows)
    dist(x, y) = sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)
    sIdx = argmin(map(x -> dist(x, imsize) , sizes))
    return sizes[sIdx]
end

function train(renderer, gtimg, lr, lossFunc; frontend="ImageView", gui=true)
    # choosing better resolution
    # @assert size(img)[1:2] == imsize "Image Sizes should be similar"
    originalSize = size(gtimg)
    imsize = sizeSelection(originalSize)
    # setup gui for visualization
    gui = imshow_gui((imsize[1]*2, imsize[2]), (1, 2))
    canvases = gui["canvas"]
    # scale ground truth image and copy to gpu
    gtimg = imresize(gtimg, imsize) .|> float32;
    gtview = reshape(
        permutedims(
            channelview(gtimg), (2, 3, 1)
        ) .|> float32,
        size(gtimg)..., 3, 1
    )|> gpu
    score = 0.0
    while score < 0.99
        CUDA.@sync preprocess(renderer)
        CUDA.@sync compactIdxs(renderer)
        CUDA.@sync forward(renderer)
        img = renderer.imageData |> cpu;
        tmpimageview = reshape(renderer.imageData, size(renderer.imageData)..., 1)
        grads = gradient(lossFunc, tmpimageview, gtview)
        CUDA.@sync ΔC = grads[1]
        CUDA.@sync backward(renderer, ΔC)
        CUDA.@sync renderer.splatData.means .-= lr*renderer.splatGrads.Δmeans
        CUDA.@sync renderer.splatData.colors .-= lr*renderer.splatGrads.Δcolors
        CUDA.@sync renderer.splatData.opacities .-= lr*renderer.splatGrads.Δopacities
        CUDA.@sync renderer.splatData.scales .-= lr*renderer.splatGrads.Δscales
        CUDA.@sync renderer.splatData.rotations .-= lr*renderer.splatGrads.Δrotations
        
        # CUDA.@sync tmpimageview .-= lr*ΔC
        yimg = colorview(RGB{N0f8},
            permutedims(
                reshape(clamp.(tmpimageview |> cpu, 0.0, 1.0), size(img)...),
                (3, 1, 2),
            ) .|> n0f8
        )
        resetGrads(renderer.splatGrads)
        imshow!(canvases[1, 2], yimg)
        imshow!(canvases[1, 1], gtimg)
    end
end

