
function compactHits(hits, bbs, hitscan, hitIdxs)
    txIdx = threadIdx().x
    tyIdx = threadIdx().y
    bxIdx = blockIdx().x
    byIdx = blockIdx().y
    bIdx = gridDim().x*(byIdx - 1i32) + bxIdx
    shmem = CuDynamicSharedArray(UInt32, (blockDim().x, blockDim().y))
    shmem[txIdx, tyIdx] = UInt32(hitscan[txIdx, tyIdx, bIdx])
    sync_threads()
    if hits[txIdx, tyIdx, bIdx] == 1
        idx = Int32(shmem[txIdx, tyIdx])
        if idx != 0
            hitIdxs[txIdx, tyIdx, idx] = bIdx
        end
    end
    sync_threads()
    return
end
