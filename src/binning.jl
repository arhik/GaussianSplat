
function hitBinning(hits, bbs, blockSizeX, blockSizeY, gridSizeX, gridSizeY)
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    #xbbmin = max(1.0f0, floor(bbs[1, 1, idx]))
    xbbmin = floor(bbs[1, 1, idx])
    #xbbmax = min(blockSize, ceil(bbs[1, 2, idx]))
    xbbmax = ceil(bbs[1, 2, idx])
    #ybbmin = min(1.0f0, floor(bbs[2, 1, idx]))
    ybbmin = floor(bbs[2, 1, idx])
    #ybbmax = max(blockSizeY, ceil(bbs[2, 2, idx]))
    ybbmax = ceil(bbs[2, 2, idx])
    # sync_threads()
    bminxIdx = Int32(div(xbbmin, float32(blockSizeX))) + 1i32
    bminyIdx = Int32(div(ybbmin, float32(blockSizeY))) + 1i32
    bmaxxIdx = Int32(div(xbbmax, float32(blockSizeX))) + 1i32
    bmaxyIdx = Int32(div(ybbmax, float32(blockSizeY))) + 1i32
    # # BB Cover 
    sync_threads()
    if bminxIdx > bmaxxIdx
        return
    elseif bminyIdx > bmaxyIdx
        return
    else
        for i in bminxIdx:bmaxxIdx
            for j in bminyIdx:bmaxyIdx
                if (1 <= i <= gridSizeX) && (1 <= j <= gridSizeY)
                    hits[i, j, idx] = 1
                end
            end
        end
    end
    sync_threads()
    return
end

function packTileId(x::UInt64)

end

function packZValue(x::UInt64)

end

function unpackTileId(x::UInt64)

end

function unpackZValue(x::UInt64)
    
end

function binPacking(bbs, packedIds, blockSizeX, blockSizeY, gridSizeX, gridSizeY)
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    xbbmin = (floor(bbs[1, 1, idx]))
    xbbmax = (ceil(bbs[1, 2, idx]))
    ybbmin = (floor(bbs[2, 1, idx]))
    ybbmax = (ceil(bbs[2, 2, idx]))
    # sync_threads()
    bminxIdx = Int32(div(xbbmin, float32(blockSizeX))) + 1i32
    bminyIdx = Int32(div(ybbmin, float32(blockSizeY))) + 1i32
    bmaxxIdx = Int32(div(xbbmax, float32(blockSizeX))) + 1i32
    bmaxyIdx = Int32(div(ybbmax, float32(blockSizeY))) + 1i32
    # # BB Cover 
    sync_threads()
end