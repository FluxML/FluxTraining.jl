module Models

using Flux
using Flux: @functor
using ModelUtils


include("./activations.jl")
include("./layers.jl")
include("./blocks.jl")
include("./heads.jl")

include("./efficientnet.jl")
include("./mobilenetv3.jl")
include("./xresnet.jl")


export xresnet18, xresnet50

end # module
