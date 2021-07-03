
struct Check
    checkfn
    name
    throw_error::Bool
    message
end


struct SanityCheckException <: Exception end

Base.show(io::IO, check::Check) = print(io, "Check(\"", check.name, "\")")


function runchecks(checks, learner)
    failed = trues(length(checks))
    for (i, check) in enumerate(checks)
        failed[i] = !check.checkfn(learner)
    end

    failedchecks = checks[failed]

    if isempty(failedchecks)
        return
    end

    println("$(length(failedchecks))/$(length(checks)) sanity checks failed:")

    for (i, check) in enumerate(failedchecks)
        println("---")
        println(i, ": ", check.name, " (", check.throw_error ? "ERROR" : "WARNING", ")")
        println()
        println(check.message)
    end

    if any(getfield.(failedchecks, :throw_error))
        throw(SanityCheckException())
    end
end


"""
    SanityCheck([checks; usedefault = true])

Callback that runs sanity [`Check`](#)s when the `Learner` is initialized.
If `usedefault` is `true`, it will run all checks in FluxTraining.CHECKS
in addition to the ones you pass in.
"""
mutable struct SanityCheck <: Callback
    checks::Vector{Check}
    checked::Bool
    function SanityCheck(checks = []; usedefault = isempty(checks))
        if usedefault
            checks = vcat(checks, CHECKS)
        end
        return new(checks, false)
    end
end


Base.show(io::IO, cb::SanityCheck) = print(io,
    "SanityCheck(", length(cb.checks), " checks)")



stateaccess(::SanityCheck) = (
    data = Read(),
    model = Read(),
    lossfn = Read(),
    optimizer = Read(),
    callbacks = Read(),
)


function on(::EpochBegin, phase::AbstractTrainingPhase, cb::SanityCheck, learner)
    if !cb.checked
        runchecks(cb.checks, learner)
        cb.checked = true
    end
end


# Checks


CheckDataIteratorTrain() =
    Check(
        "Has training data iterator",
        true,
"""
`learner` training data iterator is `nothing` (not set). You can pass it to
`Learner` in one of the following ways:

- `Learner(model, traindataiter, opt, lossfn)`
- `Learner(model, (traindataiter, valdataiter), opt, lossfn)`
- `Learner(model, (training = traindataiter,), opt, lossfn)`
"""
        ) do learner
        !isnothing(learner.data.training)
    end

CheckDataIteratorValid() =
    Check(
        "Has validation data iterator",
        false,
"""
`learner` validation data iterator is `nothing` (not set).
It is not mandatory, but you must set it to fit `ValidationPhase`s.

You can pass it to `Learner` like this:

- `Learner(model, (traindataiter, valdataiter), opt, lossfn)`

Or if you want to use training data as validation data:

- `Learner(model, (traindataiter, traindataiter), opt, lossfn)`
"""
        ) do learner
        !isnothing(learner.data.validation)
    end

CheckIteratesTuples() =
    Check(
        "Data iterators iterate over tuples",
        true,
"""
Data iterators need to be iterable and return tuples.
This means that `for (x, y) in dataiter end` works where
`(x, y)` is a pair of model inputs and outputs.
"""
        ) do learner
        try
            batch, _ = Base.iterate(learner.data.training)
            x, y = batch
        catch
            return false
        end
        return true
    end

CheckModelLossStep() =
    Check(
        "Model and loss function compatible with data",
        true,
"""
To perform the optimization step, the model and loss function need
to be compatible with the data. This means the following must work:

- `(x, y), _ = iterate(learner.data.training)`
- `ŷ = learner.model(x)`
- `loss = learner.lossfn(ŷ, y)`
"""
        ) do learner
        try
            dev = ToGPU() in learner.callbacks.cbs ? gpu : identity
            x, y = dev(iterate(learner.data.training)[1])
            ŷ = dev(learner.model)(x)
            @assert learner.lossfn(ŷ, y) isa Number
        catch
            return false
        end
        return true
    end


const CHECKS = [
    CheckDataIteratorTrain(),
    CheckDataIteratorValid(),
    CheckIteratesTuples(),
    CheckModelLossStep(),
]
