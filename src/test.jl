# Visual test for gaussian on each tile.
# means = zeros(Float32, 2, 32, 32)

# for i in axes(means, 2)
#     for j in axes(means, 3)
#         means[1, i, j] = i/size(means, 2)
#         means[2, i, j] = j/size(means, 3)
#     end
# end

# means = reshape(means, 2, n) |> CuArray
