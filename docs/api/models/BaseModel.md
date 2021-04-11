# BaseModel

Base model class.



## Parameters

- **entities**

- **relations**

- **hidden_dim**

- **scoring**

- **gamma**


## Attributes

- **embeddings**

    Extracts embeddings.

- **name**


## Examples

```python
>>> from ckb import models
>>> from ckb import scoring
>>> from mkb import datasets as mkb_datasets

>>> import torch

>>> _ = torch.manual_seed(42)

>>> dataset = mkb_datasets.CountriesS1(1)

>>> model = models.BaseModel(
...     entities = dataset.entities,
...     relations=dataset.relations,
...     hidden_dim=3,
...     gamma=3,
...     scoring=scoring.TransE(),
... )

>>> sample = torch.tensor([[3, 0, 4], [5, 1, 6]])

>>> head, relation, tail, shape = model.batch(sample)

>>> head
['belize', 'falkland_islands']

>>> tail
['morocco', 'saint_vincent_and_the_grenadines']
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **input**    
    - **kwargs**    
    
???- note "add_module"

    Adds a child module to the current module.

    The module can be accessed as an attribute using the given name.  Args:     name (string): name of the child module. The child module can be         accessed from this module using the given name     module (Module): child module to be added to the module.

    **Parameters**

    - **name**     (*str*)    
    - **module**     (*Union[ForwardRef('Module'), NoneType]*)    
    
???- note "apply"

    Applies ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self. Typical use includes initializing the parameters of a model (see also :ref:`nn-init-doc`).

    Args:     fn (:class:`Module` -> None): function to be applied to each submodule  Returns:     Module: self  Example::      >>> @torch.no_grad()     >>> def init_weights(m):     >>>     print(m)     >>>     if type(m) == nn.Linear:     >>>         m.weight.fill_(1.0)     >>>         print(m.weight)     >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))     >>> net.apply(init_weights)     Linear(in_features=2, out_features=2, bias=True)     Parameter containing:     tensor([[ 1.,  1.],             [ 1.,  1.]])     Linear(in_features=2, out_features=2, bias=True)     Parameter containing:     tensor([[ 1.,  1.],             [ 1.,  1.]])     Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     )     Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     )

    **Parameters**

    - **fn**     (*Callable[[ForwardRef('Module')], NoneType]*)    
    
???- note "batch"

    Process input sample.

    **Parameters**

    - **sample**    
    - **negative_sample**     – defaults to `None`    
    - **mode**     – defaults to `None`    
    
???- note "bfloat16"

    Casts all floating point parameters and buffers to ``bfloat16`` datatype.

    Returns:     Module: self

    
???- note "buffers"

    Returns an iterator over module buffers.

    Args:     recurse (bool): if True, then yields buffers of this module         and all submodules. Otherwise, yields only buffers that         are direct members of this module.  Yields:     torch.Tensor: module buffer  Example::      >>> for buf in model.buffers():     >>>     print(type(buf), buf.size())     <class 'torch.Tensor'> (20L,)     <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

    **Parameters**

    - **recurse**     (*bool*)     – defaults to `True`    
    
???- note "children"

    Returns an iterator over immediate children modules.

    Yields:     Module: a child module

    
???- note "cpu"

    Moves all model parameters and buffers to the CPU.

    Returns:     Module: self

    
???- note "cuda"

    Moves all model parameters and buffers to the GPU.

    This also makes associated parameters and buffers different objects. So it should be called before constructing optimizer if the module will live on GPU while being optimized.  Args:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[int, torch.device, NoneType]*)     – defaults to `None`    
    
???- note "default_batch"

???- note "distill"

    Default distillation method

    **Parameters**

    - **sample**    
    - **negative_sample**     – defaults to `None`    
    - **mode**     – defaults to `None`    
    
???- note "double"

    Casts all floating point parameters and buffers to ``double`` datatype.

    Returns:     Module: self

    
???- note "encode"

    Encode input sample, negative sample with respect to the mode.

    **Parameters**

    - **sample**    
    - **negative_sample**     – defaults to `None`    
    - **mode**     – defaults to `None`    
    
???- note "encoder"

    Encoder should be defined in the children class.

    **Parameters**

    - **e**    
    
???- note "eval"

    Sets the module in evaluation mode.

    This has any effect only on certain modules. See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`, etc.  This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.  Returns:     Module: self

    
???- note "extra_repr"

    Set the extra representation of the module

    To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.

    
???- note "float"

    Casts all floating point parameters and buffers to float datatype.

    Returns:     Module: self

    
???- note "format_sample"

    Adapt input tensor to compute scores.

    **Parameters**

    - **sample**    
    - **negative_sample**     – defaults to `None`    
    
???- note "forward"

    Compute scores of input sample, negative sample with respect to the mode.

    **Parameters**

    - **sample**    
    - **negative_sample**     – defaults to `None`    
    - **mode**     – defaults to `None`    
    
???- note "half"

    Casts all floating point parameters and buffers to ``half`` datatype.

    Returns:     Module: self

    
???- note "head_batch"

    Used to get faster when computing scores for negative samples.

    **Parameters**

    - **sample**    
    - **negative_sample**    
    
???- note "load_state_dict"

    Copies parameters and buffers from :attr:`state_dict` into this module and its descendants. If :attr:`strict` is ``True``, then the keys of :attr:`state_dict` must exactly match the keys returned by this module's :meth:`~torch.nn.Module.state_dict` function.

    Args:     state_dict (dict): a dict containing parameters and         persistent buffers.     strict (bool, optional): whether to strictly enforce that the keys         in :attr:`state_dict` match the keys returned by this module's         :meth:`~torch.nn.Module.state_dict` function. Default: ``True``  Returns:     ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:         * **missing_keys** is a list of str containing the missing keys         * **unexpected_keys** is a list of str containing the unexpected keys

    **Parameters**

    - **state_dict**     (*'OrderedDict[str, Tensor]'*)    
    - **strict**     (*bool*)     – defaults to `True`    
    
???- note "modules"

    Returns an iterator over all modules in the network.

    Yields:     Module: a module in the network  Note:     Duplicate modules are returned only once. In the following     example, ``l`` will be returned only once.  Example::      >>> l = nn.Linear(2, 2)     >>> net = nn.Sequential(l, l)     >>> for idx, m in enumerate(net.modules()):             print(idx, '->', m)      0 -> Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     )     1 -> Linear(in_features=2, out_features=2, bias=True)

    
???- note "named_buffers"

    Returns an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

    Args:     prefix (str): prefix to prepend to all buffer names.     recurse (bool): if True, then yields buffers of this module         and all submodules. Otherwise, yields only buffers that         are direct members of this module.  Yields:     (string, torch.Tensor): Tuple containing the name and buffer  Example::      >>> for name, buf in self.named_buffers():     >>>    if name in ['running_var']:     >>>        print(buf.size())

    **Parameters**

    - **prefix**     (*str*)     – defaults to ``    
    - **recurse**     (*bool*)     – defaults to `True`    
    
???- note "named_children"

    Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

    Yields:     (string, Module): Tuple containing a name and child module  Example::      >>> for name, module in model.named_children():     >>>     if name in ['conv4', 'conv5']:     >>>         print(module)

    
???- note "named_modules"

    Returns an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

    Yields:     (string, Module): Tuple of name and module  Note:     Duplicate modules are returned only once. In the following     example, ``l`` will be returned only once.  Example::      >>> l = nn.Linear(2, 2)     >>> net = nn.Sequential(l, l)     >>> for idx, m in enumerate(net.named_modules()):             print(idx, '->', m)      0 -> ('', Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     ))     1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

    **Parameters**

    - **memo**     (*Union[Set[ForwardRef('Module')], NoneType]*)     – defaults to `None`    
    - **prefix**     (*str*)     – defaults to ``    
    
???- note "named_parameters"

    Returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

    Args:     prefix (str): prefix to prepend to all parameter names.     recurse (bool): if True, then yields parameters of this module         and all submodules. Otherwise, yields only parameters that         are direct members of this module.  Yields:     (string, Parameter): Tuple containing the name and parameter  Example::      >>> for name, param in self.named_parameters():     >>>    if name in ['bias']:     >>>        print(param.size())

    **Parameters**

    - **prefix**     (*str*)     – defaults to ``    
    - **recurse**     (*bool*)     – defaults to `True`    
    
???- note "negative_encoding"

???- note "parameters"

    Returns an iterator over module parameters.

    This is typically passed to an optimizer.  Args:     recurse (bool): if True, then yields parameters of this module         and all submodules. Otherwise, yields only parameters that         are direct members of this module.  Yields:     Parameter: module parameter  Example::      >>> for param in model.parameters():     >>>     print(type(param), param.size())     <class 'torch.Tensor'> (20L,)     <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

    **Parameters**

    - **recurse**     (*bool*)     – defaults to `True`    
    
???- note "register_backward_hook"

    Registers a backward hook on the module.

    This function is deprecated in favor of :meth:`nn.Module.register_full_backward_hook` and the behavior of this function will change in future versions.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, torch.Tensor]]*)    
    
???- note "register_buffer"

    Adds a buffer to the module.

    This is typically used to register a buffer that should not to be considered a model parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state. Buffers, by default, are persistent and will be saved alongside parameters. This behavior can be changed by setting :attr:`persistent` to ``False``. The only difference between a persistent buffer and a non-persistent buffer is that the latter will not be a part of this module's :attr:`state_dict`.  Buffers can be accessed as attributes using given names.  Args:     name (string): name of the buffer. The buffer can be accessed         from this module using the given name     tensor (Tensor): buffer to be registered.     persistent (bool): whether the buffer is part of this module's         :attr:`state_dict`.  Example::      >>> self.register_buffer('running_mean', torch.zeros(num_features))

    **Parameters**

    - **name**     (*str*)    
    - **tensor**     (*Union[torch.Tensor, NoneType]*)    
    - **persistent**     (*bool*)     – defaults to `True`    
    
???- note "register_forward_hook"

    Registers a forward hook on the module.

    The hook will be called every time after :func:`forward` has computed an output. It should have the following signature::      hook(module, input, output) -> None or modified output  The input contains only the positional arguments given to the module. Keyword arguments won't be passed to the hooks and only to the ``forward``. The hook can modify the output. It can modify the input inplace but it will not have effect on forward since this is called after :func:`forward` is called.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[..., NoneType]*)    
    
???- note "register_forward_pre_hook"

    Registers a forward pre-hook on the module.

    The hook will be called every time before :func:`forward` is invoked. It should have the following signature::      hook(module, input) -> None or modified input  The input contains only the positional arguments given to the module. Keyword arguments won't be passed to the hooks and only to the ``forward``. The hook can modify the input. User can either return a tuple or a single modified value in the hook. We will wrap the value into a tuple if a single value is returned(unless that value is already a tuple).  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[..., NoneType]*)    
    
???- note "register_full_backward_hook"

    Registers a backward hook on the module.

    The hook will be called every time the gradients with respect to module inputs are computed. The hook should have the following signature::      hook(module, grad_input, grad_output) -> tuple(Tensor) or None  The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients with respect to the inputs and outputs respectively. The hook should not modify its arguments, but it can optionally return a new gradient with respect to the input that will be used in place of :attr:`grad_input` in subsequent computations. :attr:`grad_input` will only correspond to the inputs given as positional arguments and all kwarg arguments are ignored. Entries in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor arguments.  .. warning ::     Modifying inputs or outputs inplace is not allowed when using backward hooks and     will raise an error.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, torch.Tensor]]*)    
    
???- note "register_parameter"

    Adds a parameter to the module.

    The parameter can be accessed as an attribute using given name.  Args:     name (string): name of the parameter. The parameter can be accessed         from this module using the given name     param (Parameter): parameter to be added to the module.

    **Parameters**

    - **name**     (*str*)    
    - **param**     (*Union[torch.nn.parameter.Parameter, NoneType]*)    
    
???- note "requires_grad_"

    Change if autograd should record operations on parameters in this module.

    This method sets the parameters' :attr:`requires_grad` attributes in-place.  This method is helpful for freezing part of the module for finetuning or training parts of a model individually (e.g., GAN training).  Args:     requires_grad (bool): whether autograd should record operations on                           parameters in this module. Default: ``True``.  Returns:     Module: self

    **Parameters**

    - **requires_grad**     (*bool*)     – defaults to `True`    
    
???- note "save"

???- note "share_memory"

???- note "state_dict"

    Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are included. Keys are corresponding parameter and buffer names.  Returns:     dict:         a dictionary containing a whole state of the module  Example::      >>> module.state_dict().keys()     ['bias', 'weight']

    **Parameters**

    - **destination**     – defaults to `None`    
    - **prefix**     – defaults to ``    
    - **keep_vars**     – defaults to `False`    
    
???- note "tail_batch"

    Used to get faster when computing scores for negative samples.

    **Parameters**

    - **sample**    
    - **negative_sample**    
    
???- note "to"

    Moves and/or casts the parameters and buffers.

    This can be called as  .. function:: to(device=None, dtype=None, non_blocking=False)  .. function:: to(dtype, non_blocking=False)  .. function:: to(tensor, non_blocking=False)  .. function:: to(memory_format=torch.channels_last)  Its signature is similar to :meth:`torch.Tensor.to`, but only accepts floating point or complex :attr:`dtype`s. In addition, this method will only cast the floating point or complex parameters and buffers to :attr:`dtype` (if given). The integral parameters and buffers will be moved :attr:`device`, if that is given, but with dtypes unchanged. When :attr:`non_blocking` is set, it tries to convert/move asynchronously with respect to the host if possible, e.g., moving CPU Tensors with pinned memory to CUDA devices.  See below for examples.  .. note::     This method modifies the module in-place.  Args:     device (:class:`torch.device`): the desired device of the parameters         and buffers in this module     dtype (:class:`torch.dtype`): the desired floating point or complex dtype of         the parameters and buffers in this module     tensor (torch.Tensor): Tensor whose dtype and device are the desired         dtype and device for all parameters and buffers in this module     memory_format (:class:`torch.memory_format`): the desired memory         format for 4D parameters and buffers in this module (keyword         only argument)  Returns:     Module: self  Examples::      >>> linear = nn.Linear(2, 2)     >>> linear.weight     Parameter containing:     tensor([[ 0.1913, -0.3420],             [-0.5113, -0.2325]])     >>> linear.to(torch.double)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1913, -0.3420],             [-0.5113, -0.2325]], dtype=torch.float64)     >>> gpu1 = torch.device("cuda:1")     >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1914, -0.3420],             [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')     >>> cpu = torch.device("cpu")     >>> linear.to(cpu)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1914, -0.3420],             [-0.5112, -0.2324]], dtype=torch.float16)      >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)     >>> linear.weight     Parameter containing:     tensor([[ 0.3741+0.j,  0.2382+0.j],             [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)     >>> linear(torch.ones(3, 2, dtype=torch.cdouble))     tensor([[0.6122+0.j, 0.1150+0.j],             [0.6122+0.j, 0.1150+0.j],             [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

    **Parameters**

    - **args**    
    - **kwargs**    
    
???- note "train"

    Sets the module in training mode.

    This has any effect only on certain modules. See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`, etc.  Args:     mode (bool): whether to set training mode (``True``) or evaluation                  mode (``False``). Default: ``True``.  Returns:     Module: self

    **Parameters**

    - **mode**     (*bool*)     – defaults to `True`    
    
???- note "type"

    Casts all parameters and buffers to :attr:`dst_type`.

    Args:     dst_type (type or string): the desired type  Returns:     Module: self

    **Parameters**

    - **dst_type**     (*Union[torch.dtype, str]*)    
    
???- note "xpu"

    Moves all model parameters and buffers to the XPU.

    This also makes associated parameters and buffers different objects. So it should be called before constructing optimizer if the module will live on XPU while being optimized.  Arguments:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[int, torch.device, NoneType]*)     – defaults to `None`    
    
???- note "zero_grad"

    Sets gradients of all model parameters to zero. See similar function under :class:`torch.optim.Optimizer` for more context.

    Args:     set_to_none (bool): instead of setting to zero, set the grads to None.         See :meth:`torch.optim.Optimizer.zero_grad` for details.

    **Parameters**

    - **set_to_none**     (*bool*)     – defaults to `False`    
    
