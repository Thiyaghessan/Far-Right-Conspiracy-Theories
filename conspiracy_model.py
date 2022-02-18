import math
from enum import Enum
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

# 3 Types of People:
    #RADICALISED: Individuals who have fallen prey to conspiracies and spread it
    #CITIZEN: Non-Radicalised Indivuduals who can be radicalised
    #BANNED: Individuals Banned from Social Media and can no longer influence others
    #IMMUNE: Individuals who are opposed to the conspiracy theory and combat it (essentially counter radicals)
class State(Enum):
    RADICALISED = 0
    CITIZEN = 1
    BANNED = 2
    IMMUNE = 3


def number_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.state is state])


def number_radicalised(model):
    return number_state(model, State.RADICALISED)


def number_citizen(model):
    return number_state(model, State.CITIZEN)


def number_banned(model):
    return number_state(model, State.BANNED)

def number_immune(model):
    return number_state(model, State.IMMUNE)

class VirusOnNetwork(Model):
    """
    A virus model with some number of agents
    But instead of using a virus, we utilsie conspiracy theories
    """

    def __init__(
        self,
        num_nodes=10,
        avg_node_degree=3,
        initial_outbreak_size=1,
        conspiracy_spread_chance=0.3,
        conspiracy_event_frequency=0.6,
        recovery_chance=0.2,
        gain_resistance_chance=0.1,
    ):

        self.num_nodes = num_nodes
        prob = avg_node_degree / self.num_nodes
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)
        self.grid = NetworkGrid(self.G)
        self.schedule = BaseScheduler(self)
        self.initial_outbreak_size = (
            initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        )
        self.conspiracy_spread_chance = conspiracy_spread_chance
        self.conspiracy_event_frequency = conspiracy_event_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

        self.datacollector = DataCollector(
            {
                "Radicalised": number_radicalised,
                "Citizen": number_citizen,
                "Banned": number_banned,
                "Immune":number_immune
            }
        )

        # Create agents
        for i, node in enumerate(self.G.nodes()):
            a = VirusAgent(
                i,
                self,
                State.CITIZEN,
                self.conspiracy_spread_chance,
                self.conspiracy_event_frequency,
                self.recovery_chance,
                self.gain_resistance_chance,
            )
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)

        # Infect some nodes with the conspiracy
        radicalised_nodes = self.random.sample(self.G.nodes(), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(radicalised_nodes):
            a.state = State.RADICALISED

        self.running = True
        self.datacollector.collect(self)
    
    def no_radical_presence(self):
        '''
        Check if there are radicalised individuals in the network
        '''
        if number_state(self, State.RADICALISED) == 0:
            return True
    
    def no_citizens_present(self):
        '''
        Check if there are citizens (not radicals or counter radicals)
        '''
        if number_state(self, State.CITIZEN) == 0:
            return True
    
    def step(self):
        
        # Grab nodes so that we can ban some of the nodes:
        nodes = [node for node in self.grid.get_all_cell_contents()]

        # Prompt user to input node they would like to ban
        print("Input node you would like to Ban \
               \n(radicalised=red, citizen=blue, banned=green, immune=pink)")

        # Cycle through nodes and set color based on radicalisation status
        color_map = []
        for node in nodes:
            state = str(node.state)[6:]
            # print("Node {} is {}".format(node.unique_id, state))
            if  state == "RADICALISED":
                color_map.append('red')
            elif state == "CITIZEN":
                color_map.append('lightblue')
            elif state == "IMMUNE":
                color_map.append('pink')
            else:
                color_map.append('green')

        # plot network so user can make ban decision
        nx.draw(self.G, node_color=color_map, with_labels=True)
        plt.show()

        # Ban individual based on user input (must be int in range)
        ban = input()
        if ban:
            nodes[int(ban)].state = State.BANNED
            nodes[int(ban)].try_to_radicalise_neighbors()
            print("Node {} has been banned!".format(ban))
        else:
            print("No one got banned!")


        self.schedule.step()

        # collect data
        self.datacollector.collect(self)

    
    def printer(self, message):
        '''
        Same process to print the graph as observed above
        but only if the game reaches ending conditions
        '''
        print(message)
        nodes = [node for node in self.grid.get_all_cell_contents()]
        color_map = []
        for node in nodes:
            state = str(node.state)[6:]
            # print("Node {} is {}".format(node.unique_id, state))
            if  state == "RADICALISED":
                color_map.append('red')
            elif state == "CITIZEN":
                color_map.append('blue')
            elif state == "IMMUNE":
                color_map.append('pink')
            else:
                color_map.append('green')
        # plot network so user can make vaccination decision
        nx.draw(self.G, node_color=color_map, with_labels=True)
        return plt.show() 

    
    def run_model(self, n):
        '''
        Runs the model and ends prematurely if
        1. no more radicals are present
        2. no more citizens are present
        '''
        for i in range(n):
            if self.no_radical_presence():
                self.printer("There are no more radicals, it only cost you {} bans".format(number_state(self, State.BANNED)))
                break
            elif self.no_citizens_present():
                self.printer("Everyone is now Polarized")
                break
            else:
                self.step()
        self.printer("Simulation Complete, you have \
        \n{} bans, {} radicals, {} counter-radicals, {} citizens"\
              .format(number_state(self, State.BANNED), number_state(self, State.RADICALISED), \
                     number_state(self, State.IMMUNE), number_state(self, State.CITIZEN)))


class VirusAgent(Agent):
    def __init__(
        self,
        unique_id,
        model,
        initial_state,
        conspiracy_spread_chance,
        conspiracy_event_frequency,
        recovery_chance,
        gain_resistance_chance,
    ):
        super().__init__(unique_id, model)

        self.state = initial_state

        self.conspiracy_spread_chance = conspiracy_spread_chance
        self.conspiracy_event_frequency = conspiracy_event_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

    def try_to_radicalise_neighbors(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        susceptible_neighbors = [
            agent
            for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
            if agent.state is State.CITIZEN
        ]
        for a in susceptible_neighbors:
            if np.random.random() < self.conspiracy_spread_chance:
                a.state = State.RADICALISED
        
    def try_become_counter_radical(self):
        '''
        If there are radicals, some people can become counter-radicals,
        and part of the resistance
        '''
        if np.random.random() < self.gain_resistance_chance:
            self.state = State.IMMUNE

    def try_deradicalise(self):
        '''
        Sometime radicals pull themselves out of the rabbit hole
        But they can always be vulnerable to re-radicalisation
        '''
        # Try to de radicalise
        if np.random.random() < self.recovery_chance:
            # Success
            self.state = State.CITIZEN
        else:
            # Failed
            self.state = State.RADICALISED

    def try_simulate_newscycle(self):
        '''
        On days when no new events occur and the conspiracy loses steam,
        radicals can deradicalise and become citizens
        They are however, still vulnerable
        '''
        if np.random.random() < self.conspiracy_event_frequency:
            # Checking...
            if self.state is State.RADICALISED:
                self.try_deradicalise()
    
    def step(self):
        '''
        Simulate one turn
        1. Radicals try and spread conspiracies
        2. People can become counter radicals
        3. Conspiracies can lose steam and radicals stop spreading misinformation
        4. Bans if effected can radicalise individuals
        '''
        if self.state is State.RADICALISED:
            self.try_to_radicalise_neighbors()
            self.try_become_counter_radical()
        self.try_simulate_newscycle()
