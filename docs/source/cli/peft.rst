peft
====

The ``peft`` config group manages ``parameter efficient finetuning`` strategies. Note that this feature is still in beta
phase. By default (``peft=none``) no peft is performed. If activated the model will get injected some adapters that will
be trainable but all other backbone parameters will be frozen. Note that model heads always remain trainable. See
:class:`~mml.core.models.torch_base.BaseModel` for implementation details and
`huggingface/peft <https://huggingface.co/docs/peft/index>`_ for the details of the library that ``MML`` leverages.
Note that once peft is activated on a model this process is not reversible, any loading of the model will always use
the originally configured peft method. The following example gives a good overview on som eof the configuration options:

lora
~~~~

Low-Rank Adaptation (LoRA) is a PEFT method that decomposes a large matrix into two smaller low-rank matrices in the
some model layers. This drastically reduces the number of parameters that need to be fine-tuned.
Please refer to the `docs <https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig>`_ for all
config options of LoRA.

.. autoyaml:: peft/lora.yaml