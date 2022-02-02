# ======= DOPAMINE ENVIRONMENT  =============
"""
dopamine is going to act as really global environment variable with effect on all connections not the involved ones.
# TODO: make it a class like bindsnet/environement with step/render/reset.
"""

dopamine = 0


def set_dopamine(updated_dopamine):
    global dopamine
    dopamine = updated_dopamine


def get_dopamine():
    return dopamine


if __name__ == "__main__":
    print("dopamine environment")
    print("dopamine: ", get_dopamine())
    set_dopamine(1)
    print("dopamine: ", get_dopamine())
