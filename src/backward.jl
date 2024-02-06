

function backward(renderer, ΔC)
    cimage = renderer.imageData
    transmittance = renderer.transmittance
    bbs = renderer.bbs
    hitIdxs = renderer.hitIdxs
    invCov2ds = renderer.invCov2ds
    means = renderer.splatData.means
    Δmeans = renderer.splatGrads.Δmeans
    opacities = renderer.splatData.opacities
    Δopacities = renderer.splatGrads.Δopacities
    colors = renderer.splatData.colors
    Δcolors = renderer.splatGrads.Δcolors
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=(5*(reduce(*, threads))*sizeof(Float32)) splatGrads(
            cimage, 
            ΔC,
            transmittance,
            bbs,
            hitIdxs,
            invCov2ds, 
            means,
            Δmeans,
            opacities,
            Δopacities,
            colors,
            Δcolors,
        )
    end
end