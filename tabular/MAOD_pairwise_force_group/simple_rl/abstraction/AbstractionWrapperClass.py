# Python imports.
# Other imports.
from simple_rl.agents import Agent
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction

class AbstractionWrapper(Agent):

    def __init__(self, SubAgentClass=None, agent_params={}, agent=None,
                    state_abstr=None, action_abstr=None, name_ext="-abstr"):

        # Setup the abstracted agent.
        if SubAgentClass is not None:
            assert agent is None
            self.agent = SubAgentClass(**agent_params) # too confusing, not good coding style
            print("The name of the high-level agent is {}!".format(self.agent))
        else:
            assert agent is not None
            self.agent = agent
        self.action_abstr = action_abstr
        self.state_abstr = state_abstr
        
        Agent.__init__(self, name=self.agent.name + name_ext) # high-level agent

    def act(self, ground_state, reward, is_final=False, learning=True):
        '''
        Args:
            ground_state (State)
            reward (float)
        Return:
            (str)
        '''
        if self.state_abstr is not None:
            abstr_state = self.state_abstr.phi(ground_state)
        else:
            abstr_state = ground_state

        if self.action_abstr is not None:
            # TODO: the code doesnt work when the ground state is represented as an array.
            #       I think states should be wrapped as a class object with a standard interface.
            # print('ground_state=', ground_state)
            ground_action = self.action_abstr.act(self.agent, abstr_state, ground_state, reward, is_final=is_final, learning=learning)
        else:
            ground_action = self.agent.act(abstr_state, reward, is_final=is_final, learning=learning)

        return ground_action

    def reset(self):
        # Write data.
        self.agent.reset()

        if self.action_abstr is not None:
            self.action_abstr.reset()

    def end_of_episode(self):
        self.agent.end_of_episode()
        if self.action_abstr is not None:
            self.action_abstr.end_of_episode()
