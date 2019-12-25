mutable struct Learner
    data::DataBunch
    model
    opt
    loss_func
    metrics
    callbacks
end
