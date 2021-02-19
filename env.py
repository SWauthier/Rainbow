# -*- coding: utf-8 -*-
from collections import deque
import random
# import atari_py
import cv2
import torch

from active_inference.environments.darkworld import DarkWorldEnv
from wrapper import WarpFrame

class Env():
  def __init__(self, args):
    self.device = args.device
    # self.ale = atari_py.ALEInterface()
    # self.ale.setInt('random_seed', args.seed)
    # self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    # self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    # self.ale.setInt('frame_skip', 0)
    # self.ale.setBool('color_averaging', False)
    # self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    print('Making Dark World...')
    world = DarkWorldEnv(player_fov=198, close_paths=True, max_length=args.max_episode_length)
    self.world = WarpFrame(world)
    # actions = self.ale.getMinimalActionSet()
    # self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.actions = self.world.action_space
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self):
    # state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    state = self.world.observation(self.world.render(mode='rgb_array'))
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      # self.ale.act(0)  # Use a no-op after loss of life
      self.world.step(0)
    else:
      # Reset internals
      self._reset_buffer()
      # self.ale.reset_game()
      self.world.reset()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        # self.ale.act(0)  # Assumes raw action 0 is always no-op
        _, _, done, _ = self.world.step(0)
        # if self.ale.game_over():
        if done:
        #   self.ale.reset_game()
          self.world.reset()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    # self.lives = self.ale.lives()
    self.lives = 0
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    for t in range(4):
      _, r, d, _ = self.world.step(action)
      # reward += self.ale.act(self.actions.get(action))
      reward += r
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      # done = self.ale.game_over()
      done = d
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      # lives = self.ale.lives()
      lives = 0
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    # return len(self.actions)
    return self.actions.n

  def render(self):
    # cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.imshow('screen'. self.world.render())
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
