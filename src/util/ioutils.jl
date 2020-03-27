

# Loading and saving utilities

function savemodel(model, path)
    @save path model = cpu(model)
end

function loadmodel(path)
    @load path model
    return model
end

function saveweights(model, path)
    @save path weights = params(cpu(model))
end

function loadweights(path)
    @save path weights
    return weights
end
