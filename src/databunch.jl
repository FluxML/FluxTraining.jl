mutable struct DataBunch
    traindl::DataLoader
    valdl::DataLoader
    testdl::Union{Nothing, DataLoader}
    DataBunch(traindl, valdl, testdl = nothing) = new(
        traindl, valdl, testdl)
end
