
abstract type ArtifactBackend end

abstract type Artifact end
struct ImageArtifact <: Artifact data end
struct ModelArtifact <: Artifact data end
struct FileArtifact <: Artifact
    data
end

logartifact(backend, artifact, step)

"""
    log_image(backend, name, step)

Log an image to an `ArtifactBackend`.
"""
log_image(backends::Tuple, name, step) = foreach(backends) do backend
    log_image(backend, image, step)
end

"""
    log_model(backend, name, step)

Log a model to an `ArtifactBackend`.
"""
log_model(backends::Tuple, name, step) = foreach(backends) do backend
    log_model(backend, image, step)
end


struct FileArtifacts <: ArtifactBackend
    folder
end


_initartifactbackend(backend::FileArtifacts) = mkpath(backend.folder)


function log_image(backend::FileArtifacts, name, step)

end


"""
    ArtifactLogger
"""
struct ArtifactLogger <: Callback
    backends
    ArtifactLogger(backends...) = new(backends)
end


function on(::Init, ::Phase, cb::ArtifactLogger, learner)
    for backend in cb.backends
        _initartifactbackend(backends)
    end
    learner.cbstate.artifactbackends = cb.backends
end
