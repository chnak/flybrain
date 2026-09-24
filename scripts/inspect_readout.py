import flybrainer
help(flybrainer.readout)
print('---')
help(flybrainer.Decision)
print('---')
help(flybrainer.Stimulus)
print('---')
import inspect
print('Decision signature:', inspect.signature(flybrainer.Decision))
if hasattr(flybrainer.Decision, '__dataclass_fields__'):
    for f, m in flybrainer.Decision.__dataclass_fields__.items():
        print(f' field: {f}: {m.type}')
print()
print('readout signature:', inspect.signature(flybrainer.readout))