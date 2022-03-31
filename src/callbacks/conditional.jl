
# High-level interface

"""
    throttle(callback, Event, freq = 1)
    throttle(callback, Event, seconds = 1)

Throttle `Event` type for `callback` so that it is triggered either only every
`freq`'th time  or every `seconds` seconds.

## Examples

If you want to only sporadically log metrics ([`LogMetrics`](#)) or images
([`LogVisualization`](#)), `throttle` can be used as follows.

Every 10 steps:

```julia
callback = throttle(LogMetrics(TensorBoardBackend()), StepEnd, freq = 10)
learner = Learner(<args>; callbacks=[callback])
```

Or every 5 seconds:

```julia
callback = throttle(LogMetrics(TensorBoardBackend()), StepEnd, seconds = 5)
learner = Learner(<args>; callbacks=[callback])
```
"""
function throttle(callback, event::Type{<:Event}; freq = nothing, seconds = nothing)
    xor(isnothing(freq), isnothing(seconds)) || error("Pass either `every` OR `seconds`.")
    if !isnothing(freq)
        return ConditionalCallback(callback, FrequencyThrottle(freq, event))
    elseif !isnothing(seconds)
        return ConditionalCallback(callback, TimeThrottle(seconds, event))
    end
end

# Implementation

"""
    abstract type CallbackCondition

Supertype for conditions to use with [`ConditionalCallback`](#).
To implement a `CallbackCondition`, implement
[`shouldrun`](#)`(::MyCondition, event, phase)`.

See [`FrequencyThrottle`](#), [`TimeThrottle`](#) and [`throttle`](#).
"""
abstract type CallbackCondition end

"""
    ConditionalCallback(callback, condition) <: Callback

Wrapper callback that only forwards events to the wrapped callback
if [`CallbackCondition`](#) `condition` is met. See [`throttle`](#).
"""
struct ConditionalCallback <: Callback
    callback::Callback
    condition::CallbackCondition
end

stateaccess(cc::ConditionalCallback) = stateaccess(cc.callback)
init!(cc::ConditionalCallback, learner) = init!(cc.callback, learner)

function on(event::Event, phase::Phase, cb::ConditionalCallback, learner)
    if shouldrun(cb.condition, event, phase)
        on(event, phase, cb.callback, learner)
    end
end


mutable struct FrequencyThrottle <: CallbackCondition
    freq
    event
    counter
    FrequencyThrottle(f, e) = new(f, e, 1)
end

function shouldrun(c::FrequencyThrottle, event, phase)
    if typeof(event) == c.event
        if c.counter == 1
            c.counter = c.freq
            return true
        else
            c.counter -= 1
            return false
        end
    else
        return true
    end
end

mutable struct TimeThrottle <: CallbackCondition
    seconds
    event
    timer
    TimeThrottle(s, e, t = nothing) = new(s, e, t)
end


function shouldrun(c::TimeThrottle, event, phase)
    if typeof(event) == c.event
        if isnothing(c.timer) || (time() - c.timer) > c.seconds
            c.timer = time()
            return true
        else
            return false
        end
    else
        return true
    end
end


@testset "throttle" begin
    function train(cb)
        learner = testlearner(cb)
        epoch!(learner, TrainingPhase())
        return learner.cbstate.history[TrainingPhase()]
    end

    @test train(Recorder()).steps == 16
    @testset "freq" begin
        @test train(throttle(Recorder(), StepEnd, freq = 2)).steps == 8
        @test train(throttle(Recorder(), StepEnd, freq = 16)).steps == 1
    end

    if !Base.Sys.iswindows()
        @testset "seconds" begin
            @test train(throttle(Recorder(), StepEnd, seconds = 0)).steps == 16
            @test train(throttle(Recorder(), StepEnd, seconds = 10)).steps == 1
        end
    end
end
