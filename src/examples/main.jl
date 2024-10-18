
using GaussianSplat
using GaussianSplat: forward, preprocess, compactIdxs
using CUDA
using Images


# render Parameters
threads = (16, 16)
blocks = (32, 32)
imSize = (512, 512, 3)
# "C:\Users\arhik\Downloads\GaussianSplatting\GaussianSplatting\bonsai\bonsai_30000.ply"
# renderer = getRenderer(GAUSSIAN_2D, imSize, nGaussians, threads, blocks)
renderer = getRenderer(
    :GAUSSIAN_3D,
    imSize,
    threads,
    blocks,
    joinpath(
        ENV["HOMEPATH"],
        "Downloads",
        "GaussianSplatting",
        "GaussianSplatting",
        "train",
        "train_30000.ply"
    ),
);

GC.gc()
CUDA.reclaim()

tps = preprocess(renderer)
compactIdxs(renderer, threads, blocks)
forward(renderer, tps, threads, blocks)
renderer.imageData[findall((x) -> isequal(x, NaN), renderer.imageData)] .= 0.0f0
img = renderer.imageData |> Array;
tmpimageview = reshape(renderer.imageData, size(renderer.imageData)..., 1)
yimg = colorview(RGB{N0f8},
    permutedims(
        reshape(clamp.(tmpimageview |> Array, 0.0, 1.0), size(img)...),
        (3, 1, 2),
    ) .|> n0f8
)
yimg = Images.imrotate(yimg, pi / 2)
imshow(yimg)

# include("train.jl")

# windowSize = 11
# nChannels = 3
# lossFunc = getLossFunction(imSize, windowSize, nChannels)

# train(renderer, gtimg, 1e-5, lossFunc)
