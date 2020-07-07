

"""
    efficientnetb0(; usedepthwise = false, nl = relu)

"""
function efficientnetb0(; usedepthwise = false, nl = relu)
    # stage 1
    stem = conv(3, 3, 32; nl = nl, stride = 2)

    body = Chain(

        # stage 2
        mbconv(3, 32, 32 * 1, 16, nl = nl, stride = 1, usedepthwise = usedepthwise),

        # stage 3
        mbconv(3, 16, 16 * 6, 24, nl = nl, stride = 2, usedepthwise = usedepthwise),
        mbconv(3, 24, 16 * 6, 24, nl = nl, stride = 1, usedepthwise = usedepthwise),

        # stage 4
        mbconv(5, 24, 24 * 6, 40, nl = nl, stride = 2, usedepthwise = usedepthwise),
        mbconv(5, 40, 40 * 6, 40, nl = nl, stride = 1, usedepthwise = usedepthwise),

        # stage 5
        mbconv(3, 40, 40 * 6, 80, nl = nl, stride = 2, usedepthwise = usedepthwise),
        mbconv(3, 80, 80 * 6, 80, nl = nl, stride = 1, usedepthwise = usedepthwise),
        mbconv(3, 80, 80 * 6, 80, nl = nl, stride = 1, usedepthwise = usedepthwise),

        # stage 6
        mbconv(5, 80, 80 * 6, 112, nl = nl, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 112, 112 * 6, 112, nl = nl, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 112, 112 * 6, 112, nl = nl, stride = 1, usedepthwise = usedepthwise),

        # stage 7
        mbconv(5, 112, 112 * 6, 192, nl = nl, stride = 2, usedepthwise = usedepthwise),
        mbconv(5, 192, 192 * 6, 192, nl = nl, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 192, 192 * 6, 192, nl = nl, stride = 1, usedepthwise = usedepthwise),
        mbconv(5, 192, 192 * 6, 192, nl = nl, stride = 1, usedepthwise = usedepthwise),

        # stage 8
        mbconv(3, 192, 192 * 6, 320, nl = nl, stride = 1, usedepthwise = usedepthwise),

        # stage 9
        conv(1, 320, 1280, nl = nl),
    )

    return Chain(stem, body)
end
