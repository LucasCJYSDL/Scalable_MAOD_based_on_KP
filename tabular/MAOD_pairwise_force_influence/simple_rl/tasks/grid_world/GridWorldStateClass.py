''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State
import numpy as np

class GridWorldState(State):
    ''' Class for Grid World States '''

    def __init__(self, x, y):
        State.__init__(self, data=np.asarray([x, y], dtype=int))
        self.x = round(x, 5)
        self.y = round(y, 5)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y

if __name__ == '__main__':
    s_1 = GridWorldState(1,2)
    s_2 = GridWorldState(1,2)
    test = [(s_1, s_2)]
    print((GridWorldState(1,2), GridWorldState(1,3)) in test)
    print((s_1, s_2)==(GridWorldState(1,2), GridWorldState(1,2)))
