
"""
    CustomCallback(f, Event, [TPhase = Phase, access = (;)])

A callback that runs `f(learner)` every time an event of type `Event`
during a phase of type in `Phase`.

If `f` needs to access learner state, pass `access`, a named tuple
in the same form as [`stateaccess`](#).

Instead of using [`CustomCallback`](#) it is recommended to properly
implement a [`Callback`](#).

## Examples

We can get a quick idea of when a new epoch starts as follows:

```julia
cb = CustomCallback(learner -> println("New epoch!"), EpochBegin)
```
"""
mutable struct CustomCallback{E<:Event,P<:Phase} <: Callback
    f::Any
    access
end

function CustomCallback(f, E::Type{<:Event}, P::Type{<:Phase} = Phase, access = (;))
    return CustomCallback{E, P}(f, access)
end


stateaccess(cc::CustomCallback) = cc.access


function on(::E, ::P, cb::CustomCallback{E,P}, learner) where {E<:Event,P<:Phase}
    cb.f(learner)
end


@testset "CustomCallback" begin
    @testset "Basic" begin
        x = 0
        cb = CustomCallback(learner -> (x += 1;), StepBegin)
        learner = testlearner(cb)
        epoch!(learner, TrainingPhase())
        @test x == 16
    end

    @testset "Interrupt" begin
        cb = CustomCallback(Events.StepEnd, TrainingPhase) do learner
            throw(CancelFittingException("test"))
        end
        learner = testlearner(Recorder(), cb, coeff = 3)
        @test_throws CancelFittingException epoch!(learner, TrainingPhase())
        @test learner.cbstate.history[TrainingPhase()].epochs == 0
        @test learner.cbstate.history[TrainingPhase()].steps == 0
    end
end
