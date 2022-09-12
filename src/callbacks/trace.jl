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
        learner.cbstate.tracehistory = MVHistory()
    end
end

function on(::StepEnd, phase::P, traces::Traces{P}, learner) where {P<:Phase}
    step = learner.cbstate.history[phase].steps
    for (trace_name, f) in pairs(traces.preprocess)
        val = f(learner)
        push!(learner.cbstate.tracehistory, trace_name, step, val)
    end
end

@testset "Traces" begin
    cb = Traces((keya = learner -> sum(learner.step.ys),
                 keyb = learner -> sum(learner.step.ŷs)),
                ValidationPhase)
    learner = testlearner(Recorder(), cb)
    @test_nowarn fit!(learner, 1)
    @test :keya ∈ keys(learner.cbstate.tracehistory)
    @test :keyb ∈ keys(learner.cbstate.tracehistory)
end
