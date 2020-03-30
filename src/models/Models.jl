module Models

using Flux
using Flux: @functor
using ModelUtils


include("./layers.jl")
include("./heads.jl")

include("./xresnet.jl")


export xresnet18, xresnet50

end # module
