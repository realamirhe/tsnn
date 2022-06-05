from PymoNNto import Behaviour

OVERRIDABLE_SUFFIX = "_override"


class OverrideNeurons(Behaviour):
    def set_variables(self, n):
        self.set_init_attrs_as_variables(n)
        self.overridable = [
            attr for attr in dir(n) if attr.endswith(OVERRIDABLE_SUFFIX)
        ]
        if not self.overridable:
            raise AssertionError("No overridable attributes found")

    def new_iteration(self, n):
        for attr in self.overridable:
            attr_value = getattr(n, attr)
            attr_original_key = attr[: -len(OVERRIDABLE_SUFFIX)]
            if isinstance(attr_value, list):
                setattr(n, attr_original_key, attr_value[n.iteration - 1])
            else:
                setattr(n, attr_original_key, attr_value)
