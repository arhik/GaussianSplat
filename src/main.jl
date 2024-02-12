
include("cov2d.jl")
include("boundingbox.jl")
include("binning.jl")
include("compact.jl")
include("camera.jl")
include("renderer.jl")
include("projection.jl")
using WGPUgfx
# render Parameters
nGaussians = 32*32
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

include("train.jl")

windowSize = 11
nChannels = 3
lossFunc = getLossFunction(imSize, windowSize, nChannels)

train(renderer, gtimg, 1e-5, lossFunc)
