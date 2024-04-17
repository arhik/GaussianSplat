
function compactHits(hits, sortIdxs, hitscan, hitIdxs)
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    bxIdx = blockIdx().x
    byIdx = blockIdx().y
    bIdx = gridDim().x*(byIdx - 1i32) + bxIdx
    sIdx = sortIdxs[bIdx]
    shmem = CuDynamicSharedArray(UInt32, (blockDim().x, blockDim().y))
    shmem[txIdx, tyIdx] = UInt32(hitscan[txIdx, tyIdx, bIdx])
    sync_threads()
    if hits[txIdx, tyIdx, sIdx] == 1
        idx = Int32(shmem[txIdx, tyIdx])
        if idx != 0
            hitIdxs[txIdx, tyIdx, idx] = sIdx
        end
    end
    sync_threads()
    return
end
