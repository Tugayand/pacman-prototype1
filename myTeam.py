# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from game import Actions
from util import nearestPoint
import time


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='GenericAgent', second='GenericAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.state = None
        self.probability = None
        self.path = None
        self.returning_path = None
        self.defending_positions = None
        self.my_defending_positions = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.state = 'Offensive' # Offensive / Returning / Defensive / TODO return to defensive position / TODO if stalemate, search a path without being seen by the enemy
        self.probability = 1/2 # Probability of returning home after eating a power pellet or food
        self.path = [] # The path the agent took to get its current position in the enemy territory
        self.returning_path = [] # The path the agent can take to return to its territory
        self.defending_positions = [] # The positions the agent should defend, calculated based on the layout of the board
        self.patrol_position = None # The current position the agent is going to patrol, when the agent gets there it will go to the next position
        self.get_defending_positions(game_state, []) # The positions the agent should defend, calculated based on the layout of the board
        self.defending_positions = list(set(self.defending_positions)) # Remove duplicates
        self.my_defending_positions = self.get_my_defending_positions(game_state) # The positions the agent should defend, calculated based on the layout of the board
        print("Defending positions: ", self.defending_positions)
        print("My defending positions: ", self.my_defending_positions)
        CaptureAgent.register_initial_state(self, game_state)

    def state(self):
        return self.state

    def printvariables(self):
        print("I am agent: ", self.index)
        print("Start: ", self.start)
        print("State: ", self.state)
        print("Probability: ", self.probability)
        print("Path: ", self.path)
        print("Returning Path: ", self.returning_path)
        print("--------------------")

    def get_my_defending_positions(self, game_state):
        # Depending on agent index return higher or lower half of the defending positions
        indices = self.get_team(game_state)

        sorted_defending_positions = sorted(self.defending_positions, key=lambda x: x[1])

        if self.index == indices[0]:
            return sorted_defending_positions[:len(sorted_defending_positions) // 2]
        else:
            return sorted_defending_positions[len(sorted_defending_positions) // 2:]

        
            

    def get_defending_positions(self, game_state, neigbours):
        # Get all positions on width/2 -2 or +2 depending on self.red
        # Remove all positions that are walls
        # Remove all positions that have no west neigbour (that neighbour is thus a wall) if the agent is red
        # Else remove all positions that have no east neigbour (that neighbour is thus a wall) if the agent is blue
        # Remember all remaining positions that are adjacent to each other in north or south directtion (transitivity included)
        # Remove these positions from the list, these have to be analyzed separately
        # The remaining positions are the ones that the agent should defend
        # The west or east neighbours of the removed positions will be fed to the algorithm again, until all positions are removed

        if neigbours == []:
            defending_positions = []

            if self.red:
                defending_positions = [(game_state.data.layout.width // 2 - 2, y) for y in range(game_state.data.layout.height) if not game_state.data.layout.walls[game_state.data.layout.width // 2 - 2][y]]
                defending_positions = [pos for pos in defending_positions if not game_state.data.layout.walls[pos[0] - 1][pos[1]]]
            else:
                defending_positions = [(game_state.data.layout.width // 2 + 2, y) for y in range(game_state.data.layout.height) if not game_state.data.layout.walls[game_state.data.layout.width // 2 + 2][y]]
                defending_positions = [pos for pos in defending_positions if not game_state.data.layout.walls[pos[0] + 1][pos[1]]]
            
            direct_neighbours = []
            for pos in defending_positions:
                if (pos[0], pos[1] + 1) in defending_positions:
                    direct_neighbours.append((pos[0], pos[1] + 1))
                if (pos[0], pos[1] - 1) in defending_positions:
                    direct_neighbours.append((pos[0], pos[1] - 1))

            self.defending_positions += defending_positions

            # Remove the direct neighbours from the list
            self.defending_positions = [pos for pos in defending_positions if pos not in direct_neighbours]

            if len(direct_neighbours) > 0:
                self.get_defending_positions(game_state, direct_neighbours)
            else:
                self.defending_positions += defending_positions
        else:
            # We have to analyze the adjacent positions of the neigbours to the west or east
            analysis_positions = []

            # Append all positions to east or west of the neigbours
            for pos in neigbours:
                if self.red:
                    analysis_positions.append((pos[0] - 1, pos[1]))
                else:
                    analysis_positions.append((pos[0] + 1, pos[1]))
            
            # Remove all positions that respectivly have no west or east neigbour
            if self.red:
                analysis_positions = [pos for pos in analysis_positions if not game_state.data.layout.walls[pos[0] - 1][pos[1]]]
            else:
                analysis_positions = [pos for pos in analysis_positions if not game_state.data.layout.walls[pos[0] + 1][pos[1]]]

            self.defending_positions += analysis_positions

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        #print("My state is: ", self.state, self.index)

        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        #print("Agent: ", self.index, "State: ", self.state) 
        #print(self.printvariables())   

        if self.state == 'Returning':
            # Search for a path and return the first action
            return self.breadthFirstSearch(game_state).pop(0)
        
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            self.append_path(game_state, best_action)
            return best_action
        self.append_path(game_state, best_actions[0])
        return random.choice(best_actions)

    def append_path(self, game_state, action):
        # Check if current position is on own half of the board
        # If that is the case, check if next position is not on own half of the board, this means that the agent is leaving the base, this last position will be appended to the path
        # if we are not on our own half, we need to append the next position to the path
        my_pos = game_state.get_agent_position(self.index)
        pos2 = self.get_successor(game_state, action).get_agent_position(self.index)

        if self.red:
            # If I am in my base
            if my_pos[0] <= game_state.data.layout.width / 2:
                # If I am leaving my base
                if pos2[0] > game_state.data.layout.width / 2:
                    self.path.append(my_pos)
            else:
                # I am not in my base, but will I be after this action?
                if pos2[0] <= game_state.data.layout.width / 2:
                    self.path = []
                # I am not in my base and will not be after this action, append the next position to the path
                else:
                    self.path.append(my_pos)
        else:
            if my_pos[0] >= game_state.data.layout.width / 2:
                if pos2[0] < game_state.data.layout.width / 2:
                    self.path.append(my_pos)
            else:
                if pos2[0] >= game_state.data.layout.width / 2:
                    self.path = []
                else:
                    self.path.append(my_pos)

    # Goal state: position that is on our own half
    # Return: action sequence that leads home
    def breadthFirstSearch(self, game_state):
        start_state = game_state.get_agent_position(self.index)
        goal_state = self.get_goal_state(game_state)
        visited = set()
        queue = util.Queue()
        queue.push((start_state, []))

        while not queue.isEmpty():
            current_state, actions = queue.pop()

            if current_state[0] == goal_state[0]:
                return actions

            if current_state not in visited:
                visited.add(current_state)
                successors = self.get_successors(game_state, current_state)
                for successor, action in successors:
                    queue.push((successor, actions + [action]))

        return []
    
    def aStarSearch(self, game_state):
        start_state = game_state.get_agent_position(self.index)
        goal_state = self.get_goal_state(game_state)
        visited = set()
        queue = util.PriorityQueue()
        queue.push((start_state, []), 0)

        while not queue.isEmpty():
            current_state, actions = queue.pop()

            if current_state[0] == goal_state[0]:
                return actions

            if current_state not in visited:
                visited.add(current_state)
                successors = self.get_successors(game_state, current_state)
                for successor, action in successors:
                    new_actions = actions + [action]
                    cost = self.get_cost(game_state, action) + self.heuristic(successor, game_state)
                    queue.push((successor, new_actions), cost)

        return []
    
    def heuristic(self, state, game_state):
        # The horizontal distance to go to the border
        return abs(state[0] - self.get_goal_state(game_state)[0])
    
    # TODO implement this, to use aStarSearch 
    def get_cost(self, game_state, action):
        # Return the distance to the ghost summed with the distance to the border
        return 1
            

    def get_goal_state(self, game_state):
        if self.red:
            return (game_state.data.layout.width // 2 - 1, game_state.data.layout.height // 2)
        else:
            return (game_state.data.layout.width // 2, game_state.data.layout.height // 2)

    def get_successors(self, game_state, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            if not game_state.data.layout.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                successors.append((next_state, action))
        return successors

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class GenericAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        food_list = self.get_food(successor).as_list()
        def_food_list = self.get_food_you_are_defending(successor).as_list()

        "Deffensive features"

        # The amount of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        features['num_invaders'] = len(invaders)

        # Distance to the invaders

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # If we're on defense, we don't want to stop

        if action == Directions.STOP: features['stop'] = 1


        # If we're on defense, we don't want to reverse, or take into account the cost of reversing

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Do not group up with your teammates
        teammates = [successor.get_agent_state(i) for i in self.get_team(successor)]

        if len(teammates) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in teammates if a != my_state]
            if len(dists) > 0:
                features['teammate_distance'] = min(dists)

        if self.state == 'Defensive':
            if my_state.is_pacman: 
                features['on_defense'] = 0
            else:
                features['on_defense'] = 1

            # Depending on the state of your teammate:
            # - If your teammate is on defense, you should be on defense on your own defending positions
            # - If your teammate is not, you should defend all defending positions
                
            if self.patrol_position == None:
                # Go to the closest defending position
                print("I am going to the closest defending position: ", self.my_defending_positions[0])
                self.patrol_position = self.my_defending_positions[0]

            # This will be true if the agent has to defend its own defending positions    
            if True:
                if my_pos == self.patrol_position:
                    if util.flipCoin(0.5):
                        # Choose a new defending position
                        self.patrol_position = random.choice(self.my_defending_positions)
                        print("I am changing defensive position to: ", self.patrol_position)
            else:
                if my_pos == self.patrol_position:
                    if util.flipCoin(0.5):
                        self.patrol_position = self.defending_positions((self.defending_positions.index(self.patrol_position) + 1) % len(self.defending_positions))
                        print("I am changing defensive position: ", self.patrol_position)

            # Encourage the agent to stay on its patrol position
            features['distance_to_defending_positions'] = self.get_maze_distance(my_pos, self.patrol_position)
                
            return features

        "Offensive features"
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        valid_enemies = [enemy for enemy in enemies if enemy.get_position() is not None]
        capsules_list = self.get_capsules(successor)

        # Encourage the agent to eat food
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Computes distance to the nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        #if an enemy is in sight, get the actual distance to the enemy
        if len(valid_enemies) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in valid_enemies]
            features['distance_to_enemy1'] = dists[0]
            if len(dists) > 1:
                features['distance_to_enemy2'] = dists[1]
        # if an enemy is not in sight, get the noisy distance to the enemy
        else: 
            opponents = self.get_opponents(successor) # indices in the list of agents
            noisy_distances = successor.get_agent_distances()
            # get the noisy distance to the nearest enemy
            if len(noisy_distances) > 0:
                noisy_distance1 = noisy_distances[opponents[0]]
                noisy_distance2 = noisy_distances[opponents[1]]
                features['distance_to_enemy1'] = noisy_distance1
                features['distance_to_enemy2'] = noisy_distance2

        # Compute distance to the nearest power pellet
        if len(capsules_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules_list])
            features['distance_to_capsule'] = min_distance

        # Encourage the agent to eat power pellets
        features['num_capsules'] = -len(capsules_list)

        "Returning Home Decision"
        current_food = self.get_food(game_state).as_list()
        current_power_pellets = self.get_capsules(game_state)

        food_eaten = len(current_food) > len(food_list)
        power_pellet_eaten = len(current_power_pellets) > len(capsules_list)

        #print("Food eaten: ", food_eaten, "Power pellet eaten: ", power_pellet_eaten)

        # If we are on our own half of the board, we are in a defensive state, if we are winning
        if self.red:
            if game_state.get_agent_position(self.index)[0] <= game_state.data.layout.width / 2:
                if self.get_score(game_state) > 0:
                    self.state = 'Defensive'
                else:
                    self.state = 'Offensive'
            else:
                if self.get_score(game_state) > 0:
                    print("Returning, because we are winning")
                    self.state = 'Returning'
                else:
                    self.state = 'Offensive'
        else:
            if game_state.get_agent_position(self.index)[0] >= game_state.data.layout.width / 2:
                if self.get_score(game_state) > 0:
                    self.state = 'Defensive'
                else:
                    self.state = 'Offensive'
            else:
                if self.get_score(game_state) > 0:
                    print("Returning, because we are winning")
                    self.state = 'Returning'
                else:
                    self.state = 'Offensive'

        # Return Home if the distance to the nearest enemy is less than twice the distance to the nearest food
        # Return Home with a certain probability if you have eaten a power pellet or food
        # It is possible that the state is on offense, but the agent is not a pacman, so check if the agent is a pacman, if pacman only then return home
        if (not (food_eaten or power_pellet_eaten)):
            if my_state.is_pacman and (features['distance_to_enemy1'] < 1.5 * features['distance_to_food'] or features['distance_to_enemy2'] < 1.5 * features['distance_to_food']):
                print("Returning, because the enemy is close")
                self.state = 'Returning'
                return features
        else:
            if util.flipCoin(self.probability):
                print("Returning, because of coin flip")
                self.state = 'Returning'
                return features
        
        return features
    
    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'distance_to_defending_positions': -5,'invader_distance': -10, 'stop': -100, 'reverse': -2, 'teammate_distance': 0,'successor_score': 100,'distance_to_food': -1, 'distance_to_enemy1': 1, 'distance_to_enemy2': 1, 'distance_to_capsule': -1, 'num_capsules': 100}
    