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
    originalSize = size(gtimg)
    imsize = sizeSelection(originalSize)
    gtimg = imresize(gtimg, imsize) .|> float32;
    img = renderer.imageData |> cpu;
    @assert size(img)[1:2] == imsize "Image Sizes should be similar"
    #img = img/maximum(img); # TODO remove this once training script is robust
    cimg = colorview(RGB{N0f8}, permutedims(n0f8.(img), (3, 1, 2)));
    tmpimageview = reshape(img, size(cimg)..., 3, 1) |> gpu
    
    gtview = reshape(
            permutedims(
                channelview(gtimg), (2, 3, 1)
            ) .|> float32,
            size(gtimg)..., 3, 1
        )|> gpu

    ΔC = gradient(lossFunc, tmpimageview, gtview)
    gui = imshow_gui((imsize[1]*2, imsize[2]), (1, 2))
    canvases = gui["canvas"]
    score = 0.0
    while score < 0.99999
        grads = gradient(lossFunc, tmpimageview, gtview)
        ΔC = lr*grads[1] #lr is strange ... need to check grads
        #renderer.splatData.colors .-= ΔC
        tmpimageview .-= lr*ΔC
        yimg = colorview(RGB{N0f8},
            permutedims(
                reshape(clamp.(tmpimageview |> cpu, 0.0, 1.0), size(cimg)..., nChannels),
                (3, 1, 2),
            ) .|> n0f8
        )
        imshow!(canvases[1, 2], yimg)
        imshow!(canvases[1, 1], gtimg)
    end
end

