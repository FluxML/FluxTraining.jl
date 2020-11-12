using Test
using TestSetExtensions
using FluxTraining
using Animations
using Colors
using FluxTraining: EpochEnd, LearningRate, getdataiter, protect, Protected,
    ProtectedException, Read, Write, Events, Phases
using Flux: trainable

include("./testdata.jl");
