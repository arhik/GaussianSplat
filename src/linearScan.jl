# TODO implement correct implementation later along these lines.
# Prefix sum with sequential loops is good enough I think
function linearScan(hits, hitscan)
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    bxIdx = blockIdx().x
    shmem = CuDynamicSharedArray(UInt16, (blockDim().x, blockDim().y))
    shmem[txIdx, tyIdx] = 0
    sync_threads()
    scanDim = div(size(hitscan, 3), 8*(eltype(shmem) |> sizeof))
    wIdx = scanDim*(bxIdx - 1i32)
    for i in 1:scanDim
        sIdx = wIdx + i
        shmem[txIdx, tyIdx] += hits[txIdx, tyIdx, sIdx]
        hitscan[txIdx, tyIdx, sIdx] = shmem[txIdx, tyIdx]
    end
    sync_threads()
    shmem[txIdx, tyIdx] = 0
    sync_threads()
    for i in 1:bxIdx
        if i == 1
            continue
        end
        zIdx = scanDim*(i-1)
        shmem[txIdx, tyIdx] += hitscan[txIdx, tyIdx, zIdx]
    end
    sync_threads()
    for i in 1:scanDim
        if bxIdx == 1
            continue
        end
        sIdx = wIdx + i
        hitscan[txIdx, tyIdx, sIdx] += shmem[txIdx, tyIdx]
    end
    sync_threads()
    return
end

#@cuda threads=blocks blocks=(16, ) shmem=reduce(*, blocks)*sizeof(UInt16) linearScan(hits, hitScans)
