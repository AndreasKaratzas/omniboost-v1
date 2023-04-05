
# Monte Carlo Tree Search

A clean and lightly customized version of https://github.com/eleurent/rl-agents.git.

### Usage

Explanation of the agent's hyper-parameters:

1. `budget`:
    This is the "budget" of the tree search algorithm. The natural interpretation of this variable is the tree's breadth.
2. `temperature`:
    This parameter (often denoted `c`) controls the trade-off between exploration (sampling an action that hasn't been played a lot before) and exploitation (sampling the action with the highest current empirical value estimate). There is probably a theoretical value that guarantees asymptotic convergence (just like there is one for the UCB algorithm), but these are typically quite conservative, and in practice they are often tuned manually to get better performance. Intuitively, we sample the action that has the highest:
    
    $$
    E(value) + temperature * \frac{P(priorities)}{visits}
    $$

    The first term is exploitation (trust your value estimate), and the second term is exploration (sample low visit counts):
    * If $temperature = 0$, this is pure exploitation, as we greedily maximize the value estimate.
    * If $temperature = \infty$, it is pure exploration, as the contribution of the value becomes negligible, and we are basically sampling from a uniform distribution (assuming that the prior probabilities are uniform here, for simplicity).

    The order of magnitude that you want if for the exploration term:
    
    $$
    temperature * \frac{prior}{visit} \approx V_{max}
    $$ 
    
    I.e. the maximum value (sum of estimated discounted rewards) achievable in your problem. Assume that the optimal action got unlucky has a and very low value estimate (say, 0), while another (suboptimal action) has a higher value estimate, say slightly lower that the maximum value estimate. Then, the exploration bonus of the optimal action will be sufficiently large to counteract the unlucky value estimate and get a total value greater or equal to the maximum value estimate and ensure the optimal action is sampled at least once.
    
    __TLDR__: Having a temperature high enough ensures that you explore sufficiently to not discard too early the optimal action if it gets unlucky. But if the temperature is too high, the Monte Carlo Tree Search will spend too much time exploring (and favour breadth instead of depth), so your sample efficiency will suffer. A good rule of thumb should be $temperature \approx \frac{V_{max}}{\text{prior probability}(\text{optimal action})} \approx V_{max} * \text{number of actions}$, for a uniform prior policy.
3. `max_depth`:
    Following the explanation logic behind the `budget` hyper-parameter, the natural interpretation of this variable is the tree's depth.
