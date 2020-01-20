module Models

include("./layers.jl")
include("./heads.jl")

include("./selecsls.jl")
include("./xresnet.jl")


export xresnet18, xresnet50

end # module
