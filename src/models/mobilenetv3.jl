
"""
    mobilenetv3small(; usedepthwise = true)

MobileNetV3-small from [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

"""
function mobilenetv3small(; usedepthwise = true)
    head = conv(3, 3, 16; nl = hswish, stride = 2)


    return Chain(
        head,
        mbconv(3, 16, 16, 16; nl = relu, squeeze = true, stride = 2, usedepthwise = usedepthwise),
        mbconv(3, 16, 72, 24; nl = relu, squeeze = false, stride = 2, usedepthwise = usedepthwise),
        mbconv(3, 24, 88, 24; nl = relu, squeeze = false, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 24, 96, 40; nl = hswish, squeeze = true, stride = 2, usedepthwise = usedepthwise),
        mbconv(5, 40, 240, 40; nl = hswish, squeeze = true, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 40, 240, 40; nl = hswish, squeeze = true, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 40, 120, 48; nl = hswish, squeeze = true, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 48, 144, 48; nl = hswish, squeeze = true, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 48, 288, 96; nl = hswish, squeeze = true, stride = 2, usedepthwise = usedepthwise),
        mbconv(5, 96, 576, 96; nl = hswish, squeeze = true, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 96, 576, 96; nl = hswish, squeeze = true, stride = 1, usedepthwise = usedepthwise),
        Conv((1, 1), 96 => 576),
        BatchNorm(576, hswish),
        squeezeexcitation(576),
    )
end

"""
    mobilenetv3large()

MobileNetV3-large from [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

"""
function mobilenetv3large()
    head = conv(3, 3, 16; nl = hswish, stride = 2)

    # TODO: implement
    return Chain(
        head,
        mbconv(3, 16, 16, 16; nl = relu, squeeze = true, stride = 2),
        mbconv(3, 16, 72, 24; nl = relu, squeeze = false, stride = 2),
        mbconv(3, 24, 88, 24; nl = relu, squeeze = false, stride = 1),
        mbconv(5, 24, 96, 40; nl = hswish, squeeze = true, stride = 2),
        mbconv(5, 40, 240, 40; nl = hswish, squeeze = true, stride = 1),
        mbconv(5, 40, 240, 40; nl = hswish, squeeze = true, stride = 1),
        mbconv(5, 40, 120, 48; nl = hswish, squeeze = true, stride = 1),
        mbconv(5, 48, 144, 48; nl = hswish, squeeze = true, stride = 1),
        mbconv(5, 48, 288, 96; nl = hswish, squeeze = true, stride = 2),
        mbconv(5, 96, 576, 96; nl = hswish, squeeze = true, stride = 1),
        mbconv(5, 96, 576, 96; nl = hswish, squeeze = true, stride = 1),
        Conv((1, 1), 96 => 576),
        BatchNorm(576, hswish),
        squeezeexcitation(576),
        GlobalMeanPool(),
    )
end
