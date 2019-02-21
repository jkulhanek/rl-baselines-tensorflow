import common.train_wrappers as wrappers
from common.core import AbstractAgent

_registry = dict()
_agent_registry = dict()

def register_agent(id, **kwargs):
    def wrap(agent):
        _agent_registry[id] = dict(agent = agent, **kwargs)
        return agent

    return wrap

def register_trainer(id, **kwargs):
    def wrap(trainer):
        _registry[id] = dict(trainer = trainer, **kwargs)
        return trainer
    return wrap

def make_trainer(id, **kwargs):
    instance = _registry[id]['trainer'](name = id, **kwargs)

    wargs = dict(**_registry[id])
    del wargs['trainer']
    instance = wrappers.wrap(instance, **wargs).compile()
    return instance

def make_agent(id, **kwargs):
    if isinstance(id, str):
        wargs = dict(**_agent_registry[id])
        del wargs['agent']

        wargs.update(kwargs)
        instance = _agent_registry[id]['agent'](name = id, **wargs)
        return instance
    else:
        return id(**kwargs)