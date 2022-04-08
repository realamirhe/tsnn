from PymoNNto import def_dtype, Behaviour


# Hamming distance between two vectors must be zero output and prediction
# Activity 1000 20times
# 1000 / 20 => -1/50 or +1
# max_activity
# min_activity


class VoltageBaseHomeostasis(Behaviour):
    __slots__ = ["max_ta", "min_ta", "eta_ip"]
    """
        This mechanism can be used to stabilize the neurons activity.
        https://pymonnto.readthedocs.io/en/latest/Complex_Tutorial/Homeostasis/
    """

    def set_variables(self, n):
        target_act = self.get_init_attr("target_voltage", 0.05, n)
        configure = {
            "has_long_term_effect": False,
            "max_ta": target_act,
            "min_ta": -target_act,
            "eta_ip": 0.001,  # adjust_strength
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, n))
        self.eta_ip *= -1

        n.exhaustion = n.get_neuron_vec()

    def new_iteration(self, neurons):
        greater = ((neurons.v > self.max_ta) * -1).astype(def_dtype)
        smaller = ((neurons.v < self.min_ta) * 1).astype(def_dtype)

        greater *= neurons.v - self.max_ta
        smaller *= self.min_ta - neurons.v

        change = (greater + smaller) * self.eta_ip
        neurons.exhaustion += change

        if self.has_long_term_effect:
            neurons.threshold += neurons.exhaustion
        else:
            neurons.v -= neurons.exhaustion
