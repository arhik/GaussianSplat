
include("cov2d.jl")
include("boundingbox.jl")
include("binning.jl")
include("compact.jl")
include("renderer.jl")

# render Parameters
nGaussians = 32*32*8
threads = (16, 16)
blocks = (32, 32)
imSize = (512, 512, 3)
# renderer
renderer = getRenderer(GAUSSIAN_2D, imSize, nGaussians, threads, blocks)

include("train.jl")

windowSize = 11
nChannels = 3
lossFunc = getLossFunction(imSize, windowSize, nChannels)

train(renderer, gtimg, 1e-6, lossFunc)
