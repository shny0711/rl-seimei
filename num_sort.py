# 強化学習により、数値のソートを行うプログラム
# N桁の数字を昇順にソート
# s_t: N桁の数字 
# a_t: x_i, x_(i+1)を入れ替える, 確率1でa_tによるs_(t+1)が生成される
# r_t: 数列を昇順にソートできたら報酬1, それ以外の場合には -2/(N(N-1))


N = 12348719459165
x_list = [int(x) for x in str(N)]
sorted_list = sorted(x_list)

# action
def swap(x_list, i):
    x_list[i], x_list[i+1] = x_list[i+1], x_list[i]
    return x_list

def step(x_list):
    