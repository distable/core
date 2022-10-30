from munch import Munch


class Munch2(Munch):
    """
    Munch2 allows recursive dot notation assignment
    (all parents automatically created)
    e.g.:
    m = Munch2()
    m.a.b.c = 1
    print(m.a.b.c) # 1
    """
    def __getattr__(self, item):
        # Create missing
        if item not in self:
            self[item] = Munch2()
        return self[item]