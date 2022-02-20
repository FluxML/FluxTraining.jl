using Pkg
using Pollen

using FluxTraining
const PACKAGE = FluxTraining


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


DIR = mktempdir()
@info "Serving from directory $DIR"
Pollen.serve(project, DIR, lazy=false, format = Pollen.JSON())
