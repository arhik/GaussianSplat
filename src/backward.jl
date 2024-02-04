
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
