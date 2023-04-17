import numpy as np
from copy import copy
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box


class Comp(ParallelEnv):
    #TODO use action masking
    def __init__(self):
        self.gain = 1
        self.cost = -1
        # self.gain_attack = 0
        # self.gain_forage = 1
        # self.cost_attack = -2
        # self.cost_forage = -1
        self.gain_dead = 0

        self.action_dict = {0: "attack", 1: "forage", 2: "none"}

        self.num_days = 2
        self.num_life_points = 4
        self.num_actions = len(self.action_dict)

        self.possible_agents = ["player1", "player2"]
        self.agents = copy(self.possible_agents)
         
        self.action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.agents
        }

        # [days_left, # player1_life_points, # player2_life_points, # action_other]
        self.observation_spaces = {
            agent: Box(low=0, high=4, shape=(1, 4)) for agent in self.agents
        }

    def _getPD(self):
        weather_types = []
        prob_succ_range = np.arange(0.1, 1, 0.1)
        for i in range(len(prob_succ_range)):
            pS = prob_succ_range[i]
            pT = 0.8
            pP = 0
            pR = 1 - (1 - pS) ** 2

            R = (pR * self.gain) + (1 - pR) * self.cost
            S = (
                pS * (1 - pT) * self.gain
                + (1 - pS) * (1 - pT) * self.cost
                + pS * pT * (self.gain + self.cost)
                + (1 - pS) * pT * (self.cost + self.cost)
            )
            T = pT * self.gain + (1 - pT) * self.cost
            P = self.cost

            if (T > R > P > S) and (2 * R > T + S):
                weather_types.append(
                    {
                        "pR": pR,
                        "pS": pS,
                        "pT": pT,
                        "pP": pP,
                        "R": R,
                        "S": S,
                        "T": T,
                        "P": P,
                    }
                )
        return weather_types
    
    def _prob_payoff(self, player1_action, player2_action, other_life_points):
        if other_life_points != 0:
            if player1_action == 1 and player2_action == 1:
                return self.weather_type["pR"]
            elif player1_action == 1 and player2_action == 0:
                return self.weather_type["pS"]
            elif player1_action == 0 and player2_action == 1:
                return self.weather_type["pT"]
            else:
                return self.weather_type["pP"]
        else:
            if player1_action == 0:
                return self.weather_type["pP"]
            else:
                return self.weather_type["pS"]


    def _get_payoffs(self, player1_prob_success, player2_prob_success):
        player1_possible_outcome = np.random.uniform(0, 1) <= player1_prob_success
        player2_possible_outcome = np.random.uniform(0, 1) <= player2_prob_success

        if self.player1_life_points != 0 and self.player2_life_points != 0:
            if self.player1_action == self.player2_action == 1:
                if player1_possible_outcome:
                    player1_payoff, player2_payoff = self.gain, self.gain
                else:
                    player1_payoff, player2_payoff = self.cost, self.cost

            elif self.player1_action == 1 and self.player2_action == 0:
                # player1 forage success, player2 attack success
                if player1_possible_outcome and player2_possible_outcome:
                    player1_payoff, player2_payoff = self.gain + self.cost, self.gain
                # player1 forage success, player2 attack failure
                elif player1_possible_outcome and not player2_possible_outcome:
                    player1_payoff, player2_payoff = self.gain, self.cost
                # player1 forage failure, player2 attack success
                elif not player1_possible_outcome and player2_possible_outcome:
                    player1_payoff, player2_payoff = self.cost + self.cost, self.gain
                # player1 forage failure, player2 attack failure
                else:
                    player1_payoff, player2_payoff = self.cost, self.cost
            elif self.player1_action == 0 and self.player2_action == 1:
                # player1 attack success, player2 forage success
                if player1_possible_outcome and player2_possible_outcome:
                    player1_payoff, player2_payoff = self.gain, self.gain + self.cost
                # player1 attack success, player2 forage failure
                elif player1_possible_outcome and not player2_possible_outcome:
                    player1_payoff, player2_payoff = self.gain, self.cost + self.cost
                # player1 attack failure, player2 forage success
                elif not player1_possible_outcome and player2_possible_outcome:
                    player1_payoff, player2_payoff = self.cost, self.gain
                # player1 attack failure, player2 forage failure
                else:
                    player1_payoff, player2_payoff = self.cost, self.cost
            else:
                # both players attack failure 
                player1_payoff, player2_payoff = self.cost + self.cost, self.cost + self.cost
        elif self.player1_life_points != 0 and self.player2_life_points == 0:
            if self.player1_action == 1:
                if player1_possible_outcome:
                    player1_payoff, player2_payoff = self.gain, self.gain_dead
                else:
                    player1_payoff, player2_payoff = self.cost, self.gain_dead
            else:
                player1_payoff, player2_payoff = self.cost, self.gain_dead
        elif self.player1_life_points == 0 and self.player2_life_points != 0:
            if self.player2_action == 1:
                if player2_possible_outcome:
                    player1_payoff, player2_payoff = self.gain_dead, self.gain
                else:
                    player1_payoff, player2_payoff = self.gain_dead, self.cost
            else:
                player1_payoff, player2_payoff = self.gain_dead, self.cost
        else:
            player1_payoff, player2_payoff = self.gain_dead, self.gain_dead

        return player1_payoff, player2_payoff
        

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.days_left = self.num_days

        self.player1_life_points = np.random.randint(1, self.num_life_points)
        self.player2_life_points = np.random.randint(1, self.num_life_points)

        self.player1_action = 2
        self.player2_action = 2

        self.weather_type = self._getPD()[1]

        self.observation_spaces = {
            "player1": np.array([[
            self.days_left,
            self.player1_life_points,
            self.player2_life_points,
            self.player2_action
            ]]),
            "player2": np.array([[
            self.days_left,
            self.player1_life_points,
            self.player2_life_points,
            self.player1_action
            ]])
        }

        return self.observation_spaces

    def step(self, actions):
        self.player1_action = actions["player1"]
        self.player2_action = actions["player2"]

        player1_prob_success = self._prob_payoff(
            self.player1_action,
            self.player2_action,
            self.player2_life_points
        )

        player2_prob_success = self._prob_payoff(
            self.player2_action,
            self.player1_action,
            self.player1_life_points
        )

        player1_payoff, player2_payoff = self._get_payoffs(player1_prob_success, player2_prob_success)

        # update life points
        self.player1_life_points += player1_payoff
        self.player1_life_points = np.clip(self.player1_life_points, 0, self.num_life_points - 1)
        self.player2_life_points += player2_payoff
        self.player2_life_points = np.clip(self.player2_life_points, 0, self.num_life_points - 1)

        # get rewards
        rewards = {
        "player1": -1 if self.player1_life_points == 0 else 0,
        "player2": -1 if self.player2_life_points == 0 else 0,
        }   
        
        terminations = {a: False for a in self.agents}

        truncations = {a: False for a in self.agents}
        if self.days_left == 0:
            truncations = {a: True for a in self.agents}
            self.agents = [] 
        self.days_left -= 1

        infos = {a: {} for a in self.agents}

        return self.observation_spaces, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
