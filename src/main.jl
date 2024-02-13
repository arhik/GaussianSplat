
include("cov2d.jl")
include("boundingbox.jl")
include("binning.jl")
include("compact.jl")
include("camera.jl")
include("renderer.jl")
include("projection.jl")
using WGPUgfx
# render Parameters
threads = (16, 16)
blocks = (32, 32)
imSize = (512, 512, 3)
# renderer
#renderer = getRenderer(GAUSSIAN_2D, imSize, nGaussians, threads, blocks)
renderer = getRenderer(
        GAUSSIAN_3D, 
        joinpath(pkgdir(WGPUgfx), "assets", "bonsai", "bonsai_30000.ply"),
        imSize, 
        threads, 
        blocks; 
)

GC.gc()
CUDA.reclaim()

preprocess(renderer)
compactIdxs(renderer)
forward(renderer)
renderer.imageData[findall((x) -> isequal(x, NaN), renderer.imageData)] .= 0.0f0
img = renderer.imageData |> cpu;
tmpimageview = reshape(renderer.imageData, size(renderer.imageData)..., 1)
yimg = colorview(RGB{N0f8},
        permutedims(
        reshape(clamp.(tmpimageview |> cpu, 0.0, 1.0), size(img)...),
        (3, 1, 2),
        ) .|> n0f8
)
yimg = Images.imrotate(yimg, -pi/2)




include("train.jl")

windowSize = 11
nChannels = 3
lossFunc = getLossFunction(imSize, windowSize, nChannels)

train(renderer, gtimg, 1e-5, lossFunc)
