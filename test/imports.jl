using Test
using TestSetExtensions
using FluxTraining
using Animations
using Colors
using FluxTraining: EpochEnd, LearningRate, protect, Protected,
    ProtectedException, Read, Write, Events, Phases, runstep, runepoch, epoch!, step!
using FluxTraining: getcallback, setcallbacks!, replacecallback!, removecallback!, addcallback!

import FluxTraining.Phases: Phase
using Flux: trainable
using Suppressor
using FluxTraining: CHECKS, SanityCheckException
using Zygote

include("./testdata.jl");
