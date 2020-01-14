using Flux
using Flux: @functor


function convlayer(size::Tuple{Integer,Integer}, insize, outsize; stride = 1)
    return Chain(Conv(size, insize => outsize, relu; stride = stride, pad = size .÷ 2),
        BatchNorm(outsize))
end

convlayer(size::Integer, args...; kwargs...) = convlayer((size, size), args...; kwargs...)


struct SelecSLSBlock
    isfirst::Bool
    conv1
    conv2
    conv3
    conv4
    conv5
    conv6
end

@functor SelecSLSBlock

function SelecSLSBlock(insize, skip, ksize, outsize, isfirst, stride)

    SelecSLSBlock(isfirst,
        convlayer(3, insize, ksize; stride = stride),
        convlayer(1, ksize, ksize),
        convlayer(3, ksize, ksize ÷ 2),
        convlayer(1, ksize ÷ 2, ksize),
        convlayer(3, ksize, ksize ÷ 2),
        convlayer(1, 2ksize + (isfirst ? 0 : skip), outsize)
    )
end


function (sb::SelecSLSBlock)(x)#::AbstractVector)
    d1 = sb.conv1(x[1])
    d2 = sb.conv2(d1) |> sb.conv3
    d3 = sb.conv4(d2) |> sb.conv5
    if sb.isfirst
        out = sb.conv6(cat(d1, d2, d3; dims = 3))
        return (out, out)
    else
        return (sb.conv6(cat(d1, d2, d3, x[2]; dims = 3)), x[2])
    end

end

#(sb::SelecSLSBlock)(x) = [x]

function selecSLSLevel(nblocks, inkernels, kernels, outkernels)
    return Chain(SelecSLSBlock(inkernels, 0, kernels, kernels, true, 2),
        [SelecSLSBlock(kernels, kernels, kernels, kernels, false, 1) for i in 2:(nblocks - 1)]...,
        SelecSLSBlock(kernels, kernels, kernels, outkernels, false, 1),
    )
end


function selecsls(levels, head = identity)
    Chain([
        convlayer(3, 3, 32; stride = 2),
        x->[x],
        [selecSLSLevel(level...) for level in levels]...,
        first,
        head
    ]...)
end


const SELECSLSCONFIGS = Dict(
    :selecsls60 => [(2, 32, 64, 128), (3, 128, 128, 288), (4, 288, 288, 416)],
    :selecsls84 => [(2, 32, 64, 144), (5, 144, 144, 304), (6, 304, 304, 512)],
    :selecsls102 => [(4, 32, 64, 128), (5, 128, 128, 288), (7, 288, 288, 480)],
    )


selecsls60(head = identity) = selecsls(SELECSLSCONFIGS[:selecsls60], head)
selecsls84(head = identity) = selecsls(SELECSLSCONFIGS[:selecsls84], head)
selecsls102(head = identity) = selecsls(SELECSLSCONFIGS[:selecsls102], head)
