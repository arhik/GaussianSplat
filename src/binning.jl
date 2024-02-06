
function hitBinning(hits, bbs, blockSizeX, blockSizeY, gridSizeX, gridSizeY)
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    xbbmin = bbs[1, 1, idx]
    xbbmax = bbs[1, 2, idx]
    ybbmin = bbs[2, 1, idx]
    ybbmax = bbs[2, 2, idx]
    sync_threads()
    bminxIdx = Int32(div(xbbmin, float(blockSizeX))) + 1i32
    bminyIdx = Int32(div(ybbmin, float(blockSizeY))) + 1i32
    bmaxxIdx = Int32(div(xbbmax, float(blockSizeX))) + 1i32
    bmaxyIdx = Int32(div(ybbmax, float(blockSizeY))) + 1i32
    # BB Cover 
    sync_threads()
    for i in bminxIdx:bmaxxIdx
        for j in bminyIdx:bmaxyIdx
            if i <= gridSizeX && j <= gridSizeY
                hits[i, j, idx] = 1
            end
        end
    end
    sync_threads()
    return
end
