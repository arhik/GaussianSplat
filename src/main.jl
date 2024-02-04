
include("cov2d.jl")
include("boundingbox.jl")
include("binning.jl")
include("compact.jl")
include("splat.jl")
include("loss.jl")

# render Parameters
n = 32*1024
threads = (16, 16)
blocks = (32, 32)
hits = CUDA.zeros(UInt8, blocks..., n);

# SplatData
means = CUDA.rand(2, n);
rots = CUDA.rand(1, n) .- 1.0f0;
colors = CUDA.rand(3, n);
scales = 2.0f0.*CUDA.rand(2, 2, n) .- 2.0f0;
opacities = CUDA.rand(1, n);
rotMats = CUDA.rand(2, 2, n);
cov2ds = CUDA.rand(2, 2, n);
bbs = CUDA.zeros(2, 2, n);
invCov2ds = similar(cov2ds);

# gradData
Δmeans = similar(means);
Δrots = similar(rots)
Δcolors = similar(colors)
Δscales = similar(colors)
Δopacities = similar(opacities)
ΔrotMats = similar(rotMats)

# imageData
cimage = CUDA.zeros(Float32, 512, 512, 3);
transmittance = CUDA.ones(Float32, 512, 512)
sz = size(cimage)[1:2];
alpha = CUDA.zeros(sz);

include("forward.jl")
include("backward.jl")
include("train.jl")
