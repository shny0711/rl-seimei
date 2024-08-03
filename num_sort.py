# 強化学習により、数値のソートを行うプログラム
# N桁の数字を昇順にソート
# s_t: N桁の数字 
# a_t: x_i, x_(i+1)を入れ替える, 確率1でa_tによるs_(t+1)が生成される
# r_t: 数列を昇順にソートできたら報酬1, それ以外の場合には -2/(N(N-1))
from env import SortEnv
from agent import DQNAgent
from tqdm import tqdm

env = SortEnv([5,4,1,1])
agent = DQNAgent(env.N, env.N-1)
episodes = 1000
bach_size = 32

with tqdm(range(episodes)) as pbar:
    for e in pbar:
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            print("time:",time)
            print("state:",state)
            print("action:",action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_network()
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}, reward: {reward}")
                break
            if len(agent.memory) > bach_size:
                agent.replay(bach_size)