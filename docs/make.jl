using Pollen
using GraphPlot
using FluxTraining
using FilePathsBase

project = Pollen.documentationproject(FluxTraining)
Pollen.fullbuild(project, Pollen.FileBuilder(Pollen.HTML(), p"dev/"))
