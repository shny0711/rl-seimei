# 強化学習により、数値のソートを行うプログラム
# N桁の数字を昇順にソート
# s_t: N桁の数字 
# a_t: x_i, x_(i+1)を入れ替える, 確率1でa_tによるs_(t+1)が生成される
# r_t: 数列を昇順にソートできたら報酬1, それ以外の場合には -2/(N(N-1))

# 暫定値
N = 54321
x_list = [int(x) for x in str(N)]
A = 1000
B = 100
sorted_list = sorted(x_list)

# action
def swap(x_list, i):
    x_list[i], x_list[i+1] = x_list[i+1], x_list[i]
    return x_list

def step(self, act):
    change_index = act
    state = self.state
    state = swap(state, change_index)
    if state == self.sorted_list:
        self.reward = 1
        self.state = state
        return state, self.reward, True
    else:
        self.reward += -2/(N*(N-1))
        self.state = state
        return state, self.reward, False
    

def reward(x_list):
    return

def choose_action(state):
    # ε-greedy法で行動を選択
    return

# Q関数をNNによる非線形関数近似手法を用いる。
# Q(s,a)がパラメータθで表現、近似した関数をQ_θ(s,a)とする。
# Q_θ(s,a)と同一のNNアーキテクチャを持つ関数Q^hat_θ(s,a)を用意する。

def main():
    # θ、キューPの初期化、theta_bar = theta, i=0
    th = th_init
    P = []
    for i in range(10):
        t = 0
        s = [int(x) for x in str(N)]
        for t in range(10):
            act = choose_action(s)
            n_s, r, done = env.step(act)
            
            P.append([s, act, r, n_s])
            
            if i > A:
                # Q関数をθについて更新
                return
            if i % B == 0:
                # θ_bar = θ
                return
            
            if done:
                break
            
            s = n_s
