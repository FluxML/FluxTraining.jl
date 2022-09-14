"""
    Traces(preprocess[, phase])

Record a trace during `phase` by apply each pre-processing function in
`preprocess` to the [`Learner`](#) to produce a trace value.
The trace is recorded at the end of each learning step.

See [`LogTraces`](#) for logging of the trace values.
```julia
cb = Traces((loss2 = learner -> learner.step.loss^2,
             avg_gnorm = learner -> mean(map((_, g) -> norm(g), pairs(learner.step.grads))))
            TrainingPhase)
```
"""
struct Traces{P<:Phase} <: Callback
    preprocess::NamedTuple
end

function Traces(preprocess::NamedTuple, P::Type{<:Phase} = Phase)
    return Traces{P}(preprocess)
end

stateaccess(::Traces) = (cbstate = (history = Read(), tracehistory = Write()),
                         step = Read())

function init!(traces::Traces, learner)
    length(traces.preprocess) == length(unique(keys(traces.preprocess))) ||
        error("Multiple traces have the same name!")
    if !haskey(learner.cbstate, :tracehistory)
        learner.cbstate.tracehistory = DefaultDict{Phase,MVHistory}(() -> MVHistory())
    end
end

function on(::StepEnd, phase::P, traces::Traces{P}, learner) where {P<:Phase}
    step = learner.cbstate.history[phase].steps
    history = learner.cbstate.tracehistory[phase]
    for (trace_name, f) in pairs(traces.preprocess)
        val = f(learner)
        push!(history, trace_name, step, val)
    end
end

@testset "Traces" begin
    cb = Traces((keya = learner -> sum(learner.step.ys),
                 keyb = learner -> sum(learner.step.ŷs)),
                ValidationPhase)
    learner = testlearner(Recorder(), cb)
    @test_nowarn fit!(learner, 1)
    @test :keya ∈ keys(learner.cbstate.tracehistory[ValidationPhase()])
    @test :keyb ∈ keys(learner.cbstate.tracehistory[ValidationPhase()])
    @test :keya ∉ keys(learner.cbstate.tracehistory[TrainingPhase()])
    @test :keyb ∉ keys(learner.cbstate.tracehistory[TrainingPhase()])
end
