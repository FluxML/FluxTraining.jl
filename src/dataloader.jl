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
    collate_fn
    _channel_samples
    _channel_batches

    function DataLoader(dataset, batchsize; shuffle = true, threaded = true, buffersize = 4, collate_fn = identity)
        return new(dataset, batchsize, shuffle, threaded, buffersize, collate_fn, nothing, nothing)

    end
end
Base.length(dl::DataLoader) = nobs(dl.dataset) ÷ dl.batchsize

function Base.iterate(dl::DataLoader)
    # close previous channels
    (isnothing(dl._channel_samples) || isopen(dl._channel_samples)) || close(dl._channel_samples)
    (isnothing(dl._channel_batches) || isopen(dl._channel_samples)) || close(dl._channel_batches)

    # open new channels
    dl._channel_samples, dl._channel_batches = makeloaderchannels(dl.dataset,
        dl.batchsize;
        buffersize = dl.buffersize,
        shuffle = dl.shuffle,
        collate_fn = dl.collate_fn)
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


function makeloaderchannels(dataset, batchsize; buffersize = 1, shuffle = true, collate_fn = collate)
    idxs = 1:nobs(dataset)
    if shuffle
        idxs = Random.shuffle(idxs)
    end

    samplechannel = LengthChannel{Any}(nobs(dataset), buffersize * batchsize) do ch
        while true
            try
                Threads.@threads for idx in idxs
                    sample = getobs(dataset, idx)
                    put!(ch, sample)
                end
            catch e
                @error "Error in sample channel on thread $(Threads.threadid()), closing channel" error = e
                close(ch)
                throw(e)
            end
        end
    end

    batchchannel = LengthChannel{Any}(nobs(dataset) ÷ batchsize, buffersize) do ch
        while true
            try
                i = 1
                batch = Array{Any}(undef, batchsize)
                for sample in samplechannel
                    batch[i] = sample
                    if i == batchsize
                        put!(ch, batch)
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

    return samplechannel, batchchannel
end


function collate(batch)
    batch = Tuple([d[i] for d in batch] for i in 1:length(first(batch)))
    return map(catbatch, batch)
end

catbatch(xs) = cat(xs...; dims = ndims(first(xs)) + 1)
