
include("cov2d.jl")
include("boundingbox.jl")
include("binning.jl")
include("compact.jl")
include("renderer.jl")
include("loss.jl")

# render Parameters
n = 32*32*1024
threads = (16, 16)
blocks = (32, 32)

# renderer
renderer = getRenderer(GAUSSIAN_2D, (512, 512, 3), n, threads, blocks)
preprocess(renderer)
hitIdxs = compactIdxs(renderer)
forward(renderer, hitIdxs)

include("train.jl")
