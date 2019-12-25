import Flux: params, gpu
import Flux.Optimise: update!
import Zygote: Params, gradient
import IterTools: imap


function train!(model, loss, opt, data, nepochs::Integer; callbacks = [], device = gpu)
    model = model |> device
    ps = Params(params(model))
    cbhandler = CallbackHandler(callbacks)

    on_train_begin(cbhandler, nepochs, opt, data)
    for e in nepochs
        on_epoch_begin(cbhandler)

        for batch in imap(device, data)
            batch = on_batch_begin(cbhandler, batch)

            x, y = batch
            gs = gradient(ps) do
                output = model(x)
                on_loss_begin(cbhandler, output)

                lossbatch = loss(output, y)

                on_backward_begin(cbhandler, lossbatch)
                return lossbatch
            end
            on_backward_end(cbhandler, gs)

            update!(opt, ps, gs)

            on_batch_end(cbhandler)
        end

        on_epoch_end(cbhandler)
    end
    on_train_end(cbhandler, nothing)


end
