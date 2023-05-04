import numpy as np

from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box

class Comp(ParallelEnv):
    metadata = {"render_mode": ["human"]}
    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.gain = 1
        self.cost = -1
        self.gain_dead = 0
        self.num_days = 2
        self.num_life_points = 4
        
        self.action_dict = {0: "attack", 1: "forage", 2: "none"}
        self.num_actions = len(self.action_dict)

        self.possible_agents = ["player1", "player2"]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {
            agent: Box(low=0, high=4, shape=(1, 5)) for agent in self.agents
        }

        self.action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.agents
        }

    def _getPD(self):
        weather_types = []
        prob_succ_range = np.arange(0.1, 1, 0.1)
        for i in range(len(prob_succ_range)):
            pS = prob_succ_range[i]
            pT = 1
            pP = 0
            pR = 1 - (1 - pS) ** 2

            R = (pR * self.gain) + ((1 - pR) * self.cost)
            S = (pS * self.gain) + ((1 - pS) * self.cost) + pT * self.cost
            T = (pT * self.gain) + ((1 - pT) * self.cost)
            P = (pP * self.gain) + ((1 - pP) * self.cost)

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
                if player1_possible_outcome and player2_possible_outcome:
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
                # both players attack
                player1_payoff, player2_payoff = self.cost, self.cost
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

    def _get_obs(self):

        return {
            "player1": {
                "observation": np.array([[
                self.days_left,
                self.player1_life_points,
                self.player2_life_points,
                self.player1_action,
                self.player2_action
            ]]),
                "action_mask": [1, 1, 0]
            },
            "player2": {
                "observation": np.array([[
                self.days_left,
                self.player1_life_points,
                self.player2_life_points,
                self.player1_action,
                self.player2_action
            ]]),
                "action_mask": [1, 1, 0]
            }
        }
        

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.days_left = self.num_days

        self.player1_life_points = np.random.randint(1, self.num_life_points)
        self.player2_life_points = np.random.randint(1, self.num_life_points)

        self.player1_action = 2
        self.player2_action = 2

        self.weather_type = self._getPD()[1]

        if self.render_mode == "human":
            self.render_text(is_start=True)

        return self._get_obs()

    def _get_rewards(self):
        return {
            "player1": -1 if self.player1_life_points == 0 else 0,
            "player2": -1 if self.player2_life_points == 0 else 0,
        }   

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
        
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        self.days_left -= 1

        if self.render_mode == "human":
            self.render_text()

        if self.days_left == 0:
            truncations = {a: True for a in self.agents}
            terminations = {a: False for a in self.agents}
            infos = {a: {} for a in self.agents}
            
            self.agents = [] 
            return self._get_obs(), self._get_rewards(), terminations, truncations, infos

        return self._get_obs(), rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode == "human":
            self.render_text()

    def render_text(self, is_start=False):
        print(f"--Days left: {self.days_left}")
        print(f"--Current life of agent 1: {self.player1_life_points}")
        print(f"--Current life of agent 2: {self.player2_life_points}")

        if not is_start:
            print(f"--Previous action of agent 1: {self.action_dict[self.player1_action]}")
            print(f"--Previous action of agent 2: {self.action_dict[self.player2_action]}")