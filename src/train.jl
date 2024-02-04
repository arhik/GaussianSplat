
img = cimage |> cpu;
#img = img/maximum(img);
cimg = colorview(RGB{N0f8}, permutedims(n0f8.(img), (3, 1, 2)));

imshow(cimg)

img = cimage |> cpu;
#img = img/maximum(img);
cimg = colorview(RGB{N0f8}, permutedims(n0f8.(img), (3, 1, 2)));

tmpimageview = reshape(cimage, size(cimg)..., 3, 1) |> gpu

using TestImages

using ImageQualityIndexes

gtimg = imresize(testimage("coffee"), size(cimg)) .|> float32;

gtview = reshape(
        permutedims(
            channelview(gtimg), (2, 3, 1)
        ) .|> float32,
        size(gtimg)..., 3, 1
    )|> gpu
#gtview = reshape(channelview(gtimg) .|> Float32, size(gtimg)..., 3, 1) |> gpu;

windowSize = 11
nChannels = size(gtview, 3)

(kernel, cdims) = initKernel(gtview, gtview, windowSize)

ΔC = gradient(loss, tmpimageview, gtview)

using ImageView
gui = imshow_gui((512, 512))
canvas = gui["canvas"]

score = 0.0
while score < 0.99999
    score = ssimScore(tmpimageview, gtview)
    @info score
    grads = gradient(ssimLoss, tmpimageview, gtview)
    ΔC = 100.0f0*grads[1] #lr is strange ... need to check grads
    Δmeans
    yimg = colorview(RGB{N0f8},
        permutedims(
            reshape(clamp.(tmpimageview |> cpu, 0.0, 1.0), size(cimg)..., nChannels),
            (3, 1, 2),
        ) .|> n0f8
    )    
    imshow!(canvas, yimg)
end

