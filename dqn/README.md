# DQN algorithm implementation for single agent reinforcement learning tasks

## DQN algorithm (with target network, DDQN and dueling DQN improvements)

### Background: Q-Learning

At its heart, DQN is based on Q-learning, a model-free, off-policy reinforcement learning algorithm.

Model-free algorithms do not rely on a model of the environment's dynamics. They learn directly from interacting with
the environment and observing the resulting states and rewards. They do not attempt to predict what will happen next,
except implicitly through the value function.
Model-free algorithms typically learn a policy (a mapping from states to actions) or a value function (which estimates
the value of states or state-action pairs) directly from experience.

Off-policy algorithms learn about a target policy (the policy they are trying to optimize) while using a different
behavior policy to explore the environment and collect data. There is a separation between how the agent acts and what
it learns from.
For example in Q-learning when agent is learning, it might use epsilon-greedy policy. However, when agent finished
learning and is given a task, it takes action based solely on learned policy (Q-table).

Here's a quick recap of Q-learning:

- Goal: Learn a Q-function, denoted as Q(s, a), which estimates the expected cumulative future reward (also called the "
  return") for taking action a in state s and then following the optimal policy thereafter.
- Q-Table: In traditional Q-learning, a Q-table is used to store the Q-values for each state-action pair.
- Update Rule (Bellman Equation): The Q-table is updated iteratively using the Bellman equation: Q(s, a) = Q(s, a) +
  α [r + γ * max_a' Q(s', a') - Q(s, a)]
    - s: Current state
    - a: Action taken in the current state
    - r: Reward received after taking action a
    - s': Next state
    - a': Possible actions in the next state
    - γ: Discount factor (0 ≤ γ ≤ 1), which determines the importance of future rewards.
    - α: Learning rate (0 < α ≤ 1), which determines how much the Q-value is updated in each iteration.
    - max_a' Q(s', a'): The maximum Q-value for the next state, representing the expected return from following the
      optimal policy from the next state onward.

### Limitations of Q-Learning with Q-Tables:

- Curse of Dimensionality: Q-tables work well for environments with a small number of states and actions. However, in
  complex environments with high-dimensional state spaces (like images from video games), the Q-table becomes enormous
  and impractical to store and update.
- Generalization: Q-tables cannot generalize to unseen states. Each state-action pair must be explicitly visited and
  updated.

### DQN: Bridging the Gap with Deep Neural Networks

DQN overcomes these limitations by using a deep neural network to approximate the Q-function instead of a Q-table.

- Q-Network: The neural network, often called the Q-network, takes the state s as input and outputs a vector of Q-values
  for each possible action a. So, Q(s, a; θ) represents the Q-value for state s and action a as estimated by the network
  with parameters θ.
- Function Approximation: The network learns to approximate the optimal Q-function through training.

### Key Components of the DQN Algorithm:

1. Experience Replay:
    - Mechanism: DQN stores the agent's experiences in a replay buffer (also called experience replay memory). An
      experience
      tuple consists of (s, a, r, s', done), where done is a boolean indicating if the episode terminated after taking
      the
      action.
    - Purpose:
        - Breaks Correlations: Sequentially sampling experiences can lead to strong correlations between updates, making
          learning unstable. Experience replay breaks these correlations by randomly sampling mini-batches of
          experiences
          from the buffer.
        - Efficient Use of Data: Experiences can be reused multiple times for learning, improving data efficiency.

2. Target Network (Fixed Q-Targets):
    - Mechanism: DQN uses two neural networks:
        - Main Network (Online Network): This network is used to select actions during interaction with the environment.
          Its
          parameters are denoted by θ.
        - Target Network: This network is used to calculate the target Q-value in the update rule. Its parameters are
          denoted by
          θ-.
    - Purpose: Using the same network to both select and evaluate actions can lead to overestimation of Q-values and
      instability in training. The target network provides a more stable target for learning.
    - Update: The target network's parameters are periodically updated by either copying the main network's parameters
      (hard update) or slowly tracking the main network's parameters (soft update).

### The DQN Algorithm Steps:

1. Initialization:
    - Initialize the main Q-network with random weights θ.
    - Initialize the target network with the same weights as the main network: θ- = θ.
    - Initialize the replay buffer.
    - Set other hyperparameters (learning rate, discount factor, exploration rate, etc.).

2. Interaction and Experience Collection:
    - For each episode:
        - Observe the initial state s.
        - For each time step:
            - Action Selection: Select an action a based on an epsilon-greedy policy:
                - With probability ε (exploration rate), choose a random action.
                - With probability 1 - ε, choose the action with the highest Q-value according to the main network: a =
                  argmax_a Q(s, a; θ).
            - Execute Action: Take action a in the environment, receive reward r, and observe the next state s'.
            - Store Experience: Store the experience tuple (s, a, r, s', done) in the replay buffer.
            - Update the current state: s = s'.

3. Training (Learning from Replay Buffer):
    - After a certain number of steps (or at regular intervals):
        - Sample a Mini-Batch: Randomly sample a mini-batch of experiences (s, a, r, s', done) from the replay buffer.
        - Calculate Target Q-Values: For each experience in the mini-batch:
            - If the episode terminated (done is True): y = r
            - Otherwise: y = r + γ * max_a' Q(s', a'; θ-) (using the target network)
        - Calculate Loss: Compute the loss, typically the mean squared error between the predicted Q-values from the
          main network Q(s, a; θ) and the target Q-values y.
        - Update Main Network: Perform a gradient descent step to update the main network's weights θ to minimize the
          loss.

4. Target Network Update:
    - Periodically update the target network's weights θ- (either hard or soft update).

5. Repeat steps 2-4.

### Benefits of DQN:

- Handles High-Dimensional State Spaces: Neural networks can effectively handle complex, high-dimensional inputs like
  images.
- Generalization: The Q-network can generalize to unseen states, as it learns a function that maps states to Q-values.
- End-to-End Learning: DQN learns directly from raw input (e.g., pixels) to actions, without requiring hand-crafted
  features.

### DDQN (Double DQN improvement)

Decouples action selection (done by the main network) from the evaluation of that action's Q-value (done by the target
network), thus mitigating overestimation bias.

**DQN formula for target Q calculation**

Q_t(s, a) =

- r, if s' is terminal
- r + γ * max_a' Q_t(s', a'), otherwise

**DDQN formula for target Q calculation**

best_action = arg(max(Q_p(s', a')))

Q_t(s, a) =

- r, if s' is terminal
- r + γ * Q_t(s', best_action), otherwise

Why does this reduce overestimation?

In Q-learning, the max operator in the target calculation can lead to overestimation. If the main network overestimates
the Q-value of certain actions, those actions will be selected more often, and their overestimation will propagate
through the updates.

DQN with a target network helps, but it doesn't fully solve the issue because the main network (which might be
overestimating) still chooses the action used in the target.
Double DQN breaks this cycle. By using the target network to evaluate the Q-value of the action selected by the main
network, it reduces the chance of propagating overestimations. It's less likely that both the main and target networks
will overestimate the same action's value.

In summary:

- DQN with Target Network: Improves stability by introducing a delay in the target values.
- Double DQN: Builds upon this by further reducing overestimation through a decoupled action selection and evaluation
  process in the target calculation. It often leads to more accurate Q-value estimates and better performance.

### Dueling DQN

1. The Core Idea: Value and Advantage

   Instead of directly estimating the Q-value Q(s, a) for each state-action pair, Dueling DQN decomposes it into two
   separate components:

    - Value Function, V(s): This estimates the value of being in a particular state s, regardless of the specific action
      taken. It answers the question: "How good is it to be in this state in general?"
    - Advantage Function, A(s, a): This estimates the relative advantage of taking a particular action a in state s
      compared to other actions. It answers the question: "How much better is it to take this action compared to other
      actions in this state?"

   The Q-value is then reconstructed by combining these two:

   **Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))**

2. Why Separate Value and Advantage?

    - Generalization: In many situations, the value of a state might be high (or low) regardless of the action taken.
      For instance, imagine a game state where winning is almost guaranteed. The state's value is very high, no matter
      what subsequent move is made. By learning a separate value function, the agent can generalize better across
      actions, particularly in states where the value dominates the advantage.
    - Learning Efficiency: The value function V(s) can learn more efficiently because it doesn't need to learn the
      nuances of each action's impact. It only needs to learn the general value of the state.
    - Focus on Relevant Actions: The advantage function A(s, a) can focus on learning the relative merits of different
      actions in a given state. This makes it easier to discern which action is best when the differences between
      actions are subtle.

## Environments description

- cart pole (simple dqn with target network, DDQN, dueling DQN, dueling DDQN, optuna)
- flappy bird - vector of numbers as observation (simple dqn with target network, DDQN, dueling DQN, dueling DDQN,
  optuna)
- flappy bird - matrix of pixel as observation