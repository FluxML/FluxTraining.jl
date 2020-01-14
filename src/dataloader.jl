import LearnBase: getobs, nobs
import Random
using LengthChannels
import SplitApplyCombine: invert
import Base: length

mutable struct DataLoader
    dataset
    batchsize::Integer
    shuffle::Bool
    threaded::Bool
    buffersize::Integer
    transform_fn
    collate_fn

    _channel_samples
    _channel_batches

    function DataLoader(
        dataset,
        batchsize;
        shuffle = true,
        threaded = true,
        buffersize = 4,
        transform_fn = identity,
        collate_fn = collate)
        return new(dataset, batchsize, shuffle, threaded, buffersize, transform_fn, collate_fn, nothing, nothing)

    end
end
Base.length(dl::DataLoader) = nobs(dl.dataset) ÷ dl.batchsize

function Base.iterate(dl::DataLoader)
    # close previous channels
    (isnothing(dl._channel_samples) || isopen(dl._channel_samples)) || close(dl._channel_samples)
    (isnothing(dl._channel_batches) || isopen(dl._channel_samples)) || close(dl._channel_batches)

    # open new channels
    dl._channel_samples = makesamplechannel(dl)
    dl._channel_batches = makebatchchannel(dl, dl._channel_samples)

    return Base.iterate(dl._channel_batches)
end
function Base.iterate(dl::DataLoader, state)

    try
        return Base.iterate(dl._channel_batches, state)
    catch e
        @error error = e
        close(dl._channel_batches)
        close(dl._channel_samples)
        throw(e)
    end
end


function makesamplechannel(dl::DataLoader)
    idxs = 1:nobs(dl.dataset)
    if dl.shuffle
        idxs = Random.shuffle(idxs)
    end

    samplechannel = LengthChannel{Any}(nobs(dl.dataset), dl.buffersize * dl.batchsize) do ch
        while true
            Threads.@threads for idx in idxs
                try
                    sample = getobs(dl.dataset, idx)
                    sample = dl.transform_fn(sample)
                    put!(ch, sample)
                catch e
                    @error "Error in sample channel on thread $(Threads.threadid()), closing channel" error = e
                    throw(e)
                end
            end
        end
    end
    return samplechannel
end


function makebatchchannel(dl::DataLoader, samplechannel)
    batchchannel = LengthChannel{Any}(nobs(dl.dataset) ÷ dl.batchsize, dl.buffersize) do ch
        # TODO: refactor iteration logic
        # TODO: add option to drop last batch
        while true
            try
                i = 1
                batch = Array{Any}(undef, dl.batchsize)
                for sample in samplechannel
                    batch[i] = sample
                    if i == dl.batchsize
                        put!(ch, dl.collate_fn(batch))
                        i = 1
                    else
                        i += 1
                    end
                end
            catch e
                @error "Error in batch channel on thread $(Threads.threadid()), closing channel" error = e
                close(ch)
                throw(e)
            end
        end
    end
end


function collate(batch)
    batch = Tuple([d[i] for d in batch] for i in 1:length(first(batch)))
    return map(catbatch, batch)
end

catbatch(xs) = cat(xs...; dims = ndims(first(xs)) + 1)

getbatch(dl::DataLoader, startidx = 1) = dl.collate_fn(
    [dl.transform_fn(getobs(dl.dataset, idx)) for idx in startidx:dl.batchsize]
)
