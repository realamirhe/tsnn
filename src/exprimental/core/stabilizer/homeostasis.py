from PymoNNto import def_dtype, Behaviour


class Homeostasis(Behaviour):
    __slots__ = ["max_ta", "min_ta", "eta_ip"]
    """
        This mechanism can be used to stabilize the neurons activity.
        https://pymonnto.readthedocs.io/en/latest/Complex_Tutorial/Homeostasis/
    """

    def set_variables(self, n):
        target_act = self.get_init_attr("target_voltage", 0.05, n)

        self.max_ta = self.get_init_attr("max_ta", target_act, n)
        self.min_ta = self.get_init_attr("min_ta", -target_act, n)

        self.adj_strength = -self.get_init_attr("eta_ip", 0.001, n)

        n.exhaustion = n.get_neuron_vec()

    def new_iteration(self, neurons):
        greater = ((neurons.v > self.max_ta) * -1).astype(def_dtype)
        smaller = ((neurons.v < self.min_ta) * 1).astype(def_dtype)

        greater *= neurons.v - self.max_ta
        smaller *= self.min_ta - neurons.v

        change = (greater + smaller) * self.adj_strength
        neurons.exhaustion += change

        neurons.v -= neurons.exhaustion
