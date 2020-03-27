
"""
    AdaptiveMeanPool(target_out)
Adaptive mean pooling layer. 'target_out' stands for the size of one layer(channel) of output.
Irrespective of Input size and pooling window size, it returns the target output dimensions.
"""
struct AdaptiveMeanPool{N}
  target_out::NTuple{N, Int}
end

function (m::AdaptiveMeanPool)(x)
  w = size(x, 1) - m.target_out[1] + 1
  h = size(x, 2) - m.target_out[2] + 1
  pdims = PoolDims(x, (w, h); padding = (0, 0), stride = (1, 1))
  return meanpool(x, pdims)
end

"""
    AdaptiveMaxPool(target_out)
Adaptive max pooling layer. 'target_out' stands for the size of one layer(channel) of output.
Irrespective of Input size and pooling window size, it returns the target output dimensions.
"""
struct AdaptiveMaxPool{N}
  target_out::NTuple{N, Int}
end

function (m::AdaptiveMaxPool)(x)
  w = size(x, 1) - m.target_out[1] + 1
  h = size(x, 2) - m.target_out[2] + 1
  pdims = PoolDims(x, (w, h); padding = (0, 0), stride = (1, 1))
  return maxpool(x, pdims)
end

"""
  flatten(x)

Flatten `x` in all dimensions except the last/batch dimension
"""
flatten(x) = reshape(x, :, size(x)[end])


