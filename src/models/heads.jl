"""
    classificationhead(n, inchannels)

Head for convolutional networks with `inchannels` output channels.
Returns vector of size `n` with `softmax` applied.
"""
function classificationhead(n, inchannels)
    return Chain(
        AdaptiveMeanPool((1,1)),
        flatten,
        Dense(inchannels, n),
        softmax
    )
end