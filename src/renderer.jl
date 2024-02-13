include("splat.jl")

abstract type AbstractGaussianRenderer end

mutable struct GaussianRenderer2D <: AbstractGaussianRenderer
    splatData::SplatData2D
    splatGrads::SplatGrads2D
    imageData::AbstractArray{Float32, 3}
    transmittance::AbstractArray{Float32, 2}
    cov2ds::AbstractArray{Float32, 3}
    bbs::AbstractArray{Float32, 3}
    invCov2ds::AbstractArray{Float32, 3}
    nGaussians::Int
    hitIdxs::Union{Nothing, AbstractArray{UInt32, 3}}
end

@enum RendererType begin
    GAUSSIAN_2D
    GAUSSIAN_3D
    OPTIMAL_PROJECTION_3D
end

rendererType2splatType = Dict(
    GAUSSIAN_2D => SPLAT2D,
    GAUSSIAN_3D => SPLAT3D,
    OPTIMAL_PROJECTION_3D => OPTIMAL_PROJECTION_SPLAT3D
)

splatType2rendererType = Dict(
    SPLAT2D => GAUSSIAN_2D,
    SPLAT3D => GAUSSIAN_3D,
    OPTIMAL_PROJECTION_SPLAT3D => OPTIMAL_PROJECTION_3D
)

function getRenderer(rendererTypeVal::Val{GAUSSIAN_2D}, nGaussians::Int, imgSize::Tuple, threads::Tuple, blocks::Tuple)
    splatVal = typeof(rendererTypeVal).paramters[1]
    splatData = initData(splatVal, nGaussians)
    splatGrads = initGrads(splatData)
    imageData = CUDA.zeros(imgSize...) # TODO dimensions checks
    transmittance = CUDA.ones(imgSize[1:end-1])
    cov2ds = CUDA.zeros(2, 2, nGaussians);
    bbs = CUDA.zeros(2, 2, nGaussians);
    invCov2ds = CUDA.zeros(size(cov2ds));
    hitIdxs = nothing
    return GaussianRenderer2D(
        splatData,
        SplatGrads,
        imageData,
        transmittance,
        cov2ds,
        bbs,
        invCov2ds,
        nGaussians,
        hitIdxs
    )
end

function getRenderer(rendererTypeVal::Val{GAUSSIAN_2D}, path::String, imgSize::Tuple, threads::Tuple, blocks::Tuple)
    splatVal = typeof(rendererTypeVal).parameters[1]
    splatData = initData(splatVal, path)
    splatGrads = initGrads(splatData)
    imageData = CUDA.zeros(imgSize...) # TODO dimensions checks
    transmittance = CUDA.ones(imgSize[1:end-1])
    cov2ds = CUDA.zeros(2, 2, nGaussians);
    bbs = CUDA.zeros(2, 2, nGaussians);
    invCov2ds = CUDA.zeros(size(cov2ds));#similar(cov2ds);
    hitIdxs = nothing
    return GaussianRenderer2D(
        splatData,
        SplatGrads,
        imageData,
        transmittance,
        cov2ds,
        bbs,
        invCov2ds,
        nGaussians,
        hitIdxs
    )
end

getRenderer(rendererType::RendererType, nGaussians::Int, imgSize::Tuple, threads::Tuple, blocks::Tuple) = getRenderer(Val(rendererType), nGaussians::Int, imgSize::Tuple, threads::Tuple, blocks::Tuple)
getRenderer(rendererType::RendererType, path::String, imgSize::Tuple, threads::Tuple, blocks::Tuple) = getRenderer(Val(rendererType), path::String, imgSize::Tuple, threads::Tuple, blocks::Tuple)

function getRenderer(rendererTypeVal::Val{GAUSSIAN_3D}, nGaussians::Int, imgSize::Tuple, threads::Tuple, blocks::Tuple)
    splatVal = Val(SPLAT3D)
    splatData = initData(splatVal, nGaussians)
    splatGrads = initGrads(splatData)
    imageData = CUDA.zeros(imgSize...) # TODO dimensions checks
    transmittance = CUDA.ones(imgSize[1:end-1])
    cov3ds = CUDA.zeros(3, 3, nGaussians);
    cov2ds = CUDA.zeros(2, 2, nGaussians);
    bbs = CUDA.zeros(2, 2, nGaussians);
    invCov2ds = CUDA.zeros(size(cov2ds));#similar(cov2ds);
    hitIdxs = nothing
    camera = nothing
    positions = nothing
    return GaussianRenderer3D(
        splatData,
        splatGrads,
        imageData,
        positions,
        transmittance,
        cov2ds,
        cov3ds,
        bbs,
        invCov2ds,
        nGaussians,
        hitIdxs,
        camera
    )
end


function getRenderer(rendererTypeVal::Val{GAUSSIAN_3D}, path::String, imgSize::Tuple, threads::Tuple, blocks::Tuple)
    splatVal = Val(SPLAT3D)
    splatData = initData(splatVal, path)
    nGaussians = splatData.opacities |> length
    splatGrads = initGrads(splatData)
    imageData = CUDA.zeros(imgSize...) # TODO dimensions checks
    transmittance = CUDA.ones(imgSize[1:end-1])
    cov3ds = CUDA.zeros(3, 3, nGaussians);
    cov2ds = CUDA.zeros(2, 2, nGaussians);
    bbs = CUDA.zeros(2, 2, nGaussians);
    invCov2ds = CUDA.zeros(size(cov2ds)); #similar(cov2ds);
    hitIdxs = nothing
    camera = nothing
    positions = nothing
    return GaussianRenderer3D(
        splatData,
        splatGrads,
        imageData,
        positions,
        transmittance,
        cov2ds,
        cov3ds,
        bbs,
        invCov2ds,
        nGaussians,
        hitIdxs,
        camera
    )
end


function getRenderer(
        rendererType::RendererType, 
        imgSize::NTuple{N, Int64}, 
        nGaussians::Int, 
        threads::Tuple, 
        blocks::Tuple;
        path::Union{Nothing, String}=nothing 
    ) where N
    rendererTypeVal = rendererType |> Val
    return getRenderer(rendererTypeVal, imgSize, nGaussians, threads, blocks; path=path)
end


Base.show(io::IO, ::MIME"text/plain", renderer::GaussianRenderer2D) = begin
    println("GaussianRenderer2D \n : $(size(renderer.imageData)) \n : $(renderer.nGaussians)")
end


# GaussianRenderer 3D version

mutable struct GaussianRenderer3D <: AbstractGaussianRenderer
    splatData::SplatData3D
    splatGrads::SplatGrads3D
    imageData::AbstractArray{Float32, 3}
    positions::Union{Nothing, AbstractArray{Float32, 2}}
    transmittance::AbstractArray{Float32, 2}
    cov2ds::AbstractArray{Float32, 3}
    cov3ds::AbstractArray{Float32, 3}
    bbs::AbstractArray{Float32, 3}
    invCov2ds::AbstractArray{Float32, 3}
    nGaussians::Int
    hitIdxs::Union{Nothing, AbstractArray{UInt32, 3}}
    camera::Union{Nothing, Camera}
end

Base.show(io::IO, ::MIME"text/plain", renderer::GaussianRenderer3D) = begin
    println("GaussianRenderer3D \n : $(size(renderer.imageData)) \n : $(renderer.nGaussians)")
end


# Forward and Backward functions are place in "forward.jl" and "backward.jl" files

include("forward.jl")
include("backward.jl")