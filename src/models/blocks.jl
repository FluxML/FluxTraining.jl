"""
    conv(ksize, k_in, k_out; nl = relu, stride = 1)

Basic convolutional module with batch normalization.

input shape: `(h, w, k_in, b)`

output shape: `(h/stride, w/stride, k_out, b)`

# Arguments

- `ksize`: kernel size
- `k_in`: number of input kernels
- `k_out`: number of output kernels

# Keyword arguments

- `nl`: non-linearity/activation function (applied after BN)
- `stride`: stride of the convolution
"""
function conv(ksize::Int, k_in::Int, k_out::Int; nl = relu, stride::Int = 1)
    return Chain(
        Conv((ksize, ksize), k_in => k_out; pad = ksize ÷ 2, stride = stride),
        BatchNorm(k_out, nl),
    )
end


"""
    mbconv(
        ksize, k_in, k_exp, k_out;
        nl = relu, stride = 1, squeeze = false, usedepthwise = true)

Mobile Inverted Bottleneck block, as described in
[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).

Also used in [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946v3)

# Arguments

- `ksize`: kernel size of expansion block
- `k_in`: number of input channels
- `k_exp`: number of channels after expansion
- `k_out`: number of output channels

# Keyword Arguments

- `nl = relu`: non-linearity/activation function to use in expansion block
- `stride`: stride of the expansion block
- `squeeze = true`: whether to add a squeeze-excitation block between expansion
  and projection
- `usedepthwise = true`: Whether to use a depthwise separable convolution as
  the expansion block. If `false`, uses a regular convolution. This option
  exists because, as of `Flux@0.10.4`, the `DepthwiseConv` layer does not run
  on GPU, see [Flux.jl Issue #459](https://github.com/FluxML/Flux.jl/issues/459)
"""
function mbconv(
        ksize,
        k_in,
        k_exp,
        k_out;
        nl = relu,
        stride = 1,
        squeeze = true,
        usedepthwise = true)

    convblock = usedepthwise ? depthwiseseparable : conv

    block = Chain(
        # expansion
        convblock(ksize, k_in, k_exp; nl = nl, stride = stride),
        squeeze ? squeezeexcitation(k_exp) : identity,

        # projection
        Conv((1, 1), k_exp => k_out),
    )

    # add residual connection if input and output are the same size
    # as done in MobileNetV3
    if k_in == k_out && stride == 1
        block = Residual(block)
    end

    return block
end



"""
    squeezeexcitation(k; ratio = 4; k_mid = k ÷ ratio, ratio = 4)

Squeeze-and-excitation block.

input shape: `(h, w, k, b)`

output shape: `(1, 1, k, b)`

# Arguments
- `k`: number of input kernels and output kernels

# Keyword arguments
- `k_mid = k ÷ ratio`: number of intermediate layers
- `ratio = 4`: ratio to calculate `k_mid`

# Notes

In [MobileNetV3], the activations are `relu` and `hardσ`, while in EfficientNet,
they are `swish` and `σ`.
"""
function squeezeexcitation(k; ratio = 4, k_mid = k ÷ ratio, nlmid = relu, nlend = hardσ)
    k_mid = k ÷ ratio

    return SkipConnection(
        Chain(
            GlobalMeanPool(),
            Conv((1, 1), k => k_mid),
            BatchNorm(k_mid, nlmid),
            Conv((1, 1), k_mid => k),
            BatchNorm(k, nlend),
        ),
        (x, weights) -> x .* weights
    )
end


function depthwiseseparable(ksize, k_in, k_exp; nl = relu, stride = 1)
    return Chain(
        Conv((1, 1), k_in => k_exp),
        BatchNorm(k_exp, nl),

        DepthwiseConv(
            (ksize, ksize),
            k_exp => k_exp,
            nl;
            pad = ksize ÷ 2,
            stride = stride
        ),
        BatchNorm(k_exp, nl)
    )
end
