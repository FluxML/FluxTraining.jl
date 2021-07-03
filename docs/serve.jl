using FluxTraining
using Pollen
using GraphPlot


project = Pollen.documentationproject(FluxTraining;
    refmodules = [FluxTraining, FluxTraining.Phases], watchpackage=true)

##

Pollen.serve(project)
