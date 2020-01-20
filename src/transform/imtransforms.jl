using ImageTransformations
using CoordinateTransformations

"""
    randomcrop((h, w), (k, l))

Get a transformation for random-cropping from size h x w to k x l.

First scale the image down then translate it randomly within the bounds so that
it can be cropped after.
"""
function randomcrop((h, w), (k, l))
    factor = max(k/h, l/w)
    scaletfm = Scale(factor, factor)
    h_, w_ = ((h, w) .* factor)
    translatetfm = Translation(-rand(0:h_-k), -rand(0:w_-l))

    return translatetfm ∘ scaletfm
end
