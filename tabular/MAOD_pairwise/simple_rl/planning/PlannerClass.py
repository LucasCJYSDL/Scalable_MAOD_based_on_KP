class Planner(object):
    ''' Abstract class for a Planner. '''

    def __init__(self, mdp, agent_id, name="planner"):

        self.name = name

        # MDP components.
        self.mdp = mdp
        self.agent_id = agent_id
        self.init_state = self.mdp.get_init_states()[agent_id]
        print("The init_state of {} is {}!".format(self.agent_id, self.init_state))
        self.states = set([])
        self.actions = mdp.get_actions()
        self.reward_func = mdp.get_reward_func_single()
        self.transition_func = mdp.get_transition_func_single()
        self.gamma = mdp.gamma
        self.has_planned = False

    def plan(self, state):
        pass

    def policy(self, state):
        pass

    def __str__(self):
        return self.name