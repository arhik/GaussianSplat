

function backward(renderer, hitIdxs)
    cimage = renderer.imageData
    transmittance = renderer.transmittance
    bbs = renderer.bbs
    invCov2ds = renderer.invCov2ds
    means = renderer.splatData.means
    Δmeans = renderer.splatGrads.Δmeans
    opacities = renderer.splatData.opacities
    Δopacities = renderer.splatGrads.Δopacities
    colors = renderer.splatData.colors
    Δcolors = renderer.splatGrads.Δcolors
    @cuda threads=threads blocks=blocks shmem=(5*(reduce(*, threads))*sizeof(Float32)) splatGrads(
        cimage, 
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