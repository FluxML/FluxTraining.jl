using Pkg
using Pollen

using GraphPlot
using FluxTraining
const PACKAGE = FluxTraining


# Create target folder
DIR = abspath(mkpath(ARGS[1]))


# Create Project
m = PACKAGE
ms = [PACKAGE,]


@info "Creating project..."
project = Project(
    Pollen.Rewriter[
        Pollen.DocumentFolder(pkgdir(m), prefix = "documents"),
        Pollen.ParseCode(),
        Pollen.ExecuteCode(),
        Pollen.PackageDocumentation(ms),
        Pollen.DocumentGraph(),
        Pollen.SearchIndex(),
        Pollen.SaveAttributes((:title,)),
        Pollen.LoadFrontendConfig(pkgdir(m))
    ],
)

@info "Rewriting documents..."
Pollen.rewritesources!(project)

@info "Writing to disk at \"$DIR\"..."
builder = Pollen.FileBuilder(
    Pollen.JSON(),
    DIR,
)
Pollen.build(
    builder,
    project,
)
