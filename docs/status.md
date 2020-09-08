
## ToDos

### Features

#### Changes

- add default optimizer and use weight decay

#### Interfaces

- Tools for safe callbacks. See [Zulip discussion](https://julialang.zulipchat.com/#narrow/stream/237432-ml-ecosystem-coordination/topic/FluxTraining.2Ejl.20%28prev.2E.20New.20here.20.3A%29.20%29/near/209066325)
    - non-writing callbacks
    - write-restricted callbacks
    - "unsafe" callbacks
- More general, extensible `Hyperparameter` API that is not restricted
  to optimizer hyperparameters. See [Zulip discussion](https://julialang.zulipchat.com/#narrow/stream/237432-ml-ecosystem-coordination/topic/FluxTraining.2Ejl.20%28prev.2E.20New.20here.20.3A%29.20%29/near/209080709)

#### Callbacks

- `ToTorch` callback that uses `Torch.jl` (@requires)
