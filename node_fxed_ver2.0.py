#ノードの固定化
import numpy as np
import math as ma

#入力
q_in1 = np.array([1,0])
q_in2 = np.array([1,0])
q_in3 = np.array([1,0])
#|q1 q2 q3>

#出力
q_out1_0 = np.array([1,0])
q_out2_0 = np.array([1,0])
q_out3_0 = np.array([1,0])
#|q1 q2 q3>

q_out1_1 = np.array([1,0])
q_out2_1 = np.array([1,0])
q_out3_1 = np.array([0,1])

q_out1_2 = np.array([1,0])
q_out2_2 = np.array([0,1])
q_out3_2 = np.array([1,0])

q_out1_3 = np.array([1,0])
q_out2_3 = np.array([0,1])
q_out3_3 = np.array([0,1])

q_out1_4 = np.array([0,1])
q_out2_4 = np.array([1,0])
q_out3_4 = np.array([1,0])

q_out1_5 = np.array([0,1])
q_out2_5 = np.array([1,0])
q_out3_5 = np.array([0,1])

q_out1_6 = np.array([0,1])
q_out2_6 = np.array([0,1])
q_out3_6 = np.array([1,0])

q_out1_7 = np.array([0,1])
q_out2_7 = np.array([0,1])
q_out3_7 = np.array([0,1])


#量子回路の定義
root2 = ma.sqrt(2)
h1 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#1
h2 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#2
h3 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#3
h4 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#1
h5 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#2
h6 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#3

CZ_23 = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,-1]]).reshape(2,2,2,2)

x_gate = np.array([[0,1],
                   [1,0]])

z_gate = np.array([[1,0],
                   [0,-1]])


i11_fixed0 = (1/root2) * np.array([1,0])
i11_fixed1 = (1/root2) * np.array([0,1])

#i33を消去(1)
i33_delete1 = np.einsum('ij,kjlm,mn -> kiln',x_gate,CZ_23,h6)

#i21を消去(1)
i21_delete1 = np.einsum('ij,jk,kopq,pr -> iorq',h2,z_gate,i33_delete1,h5)

#i31を消去(1)
i31_delete1 = np.einsum('ij ,jk, lkmn -> limn',h3,z_gate,i21_delete1)

#i14を消去
i14_delete = np.einsum('ij,jk -> ik',x_gate,h4)

#1bit目のみ縮約(1)
re1_e_1 = np.einsum('i,ij -> j',i11_fixed1,i14_delete)#rank1のテンソル

#(2,3bit目について)入力と縮約する、出力との縮約は最後(1)
re1_1 = np.einsum('i,j,ijlm -> lm',q_in2,q_in3,i31_delete1)#00/rank2のテンソル

#re1_e_1とre1_1で直積をとる
re1_1_dot = np.array([ ])
for i in range(len(re1_e_1)):#0,1
    for j in range(len(re1_1)):
        for k in range(len(re1_1)):
            #print(i,j,k)
            re1_1_dot = np.insert(re1_1_dot,len(re1_1_dot),re1_e_1[i] * re1_1[j][k])

print(re1_1_dot)

#i33を消去(0)
i33_delete0 = np.einsum('ij,kjlm,mn -> kiln',x_gate,CZ_23,h6)
#print(d_delete0)

#i21を消去(0)
i21_delete0 = np.einsum('ij,jopq,pr -> iorq',h2,i33_delete0,h5)
#print(b_delete0)

#i31を消去(0)
i31_delete0 = np.einsum('ij , kjlm -> kilm',h3,i21_delete0)

#i14の部分のみで縮約(0)
re0_e_1 = np.einsum('i,ij -> j',i11_fixed0,i14_delete)#rank1のテンソル

#入力と縮約する、出力との縮約は最後(0)
re0_1 = np.einsum('i,j,ijlm -> lm',q_in2,q_in3,i31_delete0)

#re1_e_1とre1_1で直積をとる
re0_1_dot = np.array([ ])
for i in range(len(re0_e_1)):#0,1
    for j in range(len(re0_1)):
        for k in range(len(re0_1)):
            #print(i,j,k)
            re0_1_dot = np.insert(re0_1_dot,len(re0_1_dot),re0_e_1[i] * re0_1[j][k])

#print(re0_1_dot)

#0と1で出来た直積を足す
dot_1 = np.array([ ])
for i in range(len(re0_1_dot)):
    dot_1 = np.insert(dot_1,len(dot_1),re0_1_dot[i] + re1_1_dot[i])
dot_1.reshape(2,2,2)#rank3のテンソルを生成

print("input -> |000>")
a1 = np.abs(np.einsum("ijk,i,j,k -> ",dot_1.reshape(2,2,2),q_out1_0,q_out2_0,q_out3_0) )**2#* 1/root2)**2
print('re1(|000>) = ',a1)
a2 = np.abs(np.einsum("ijk,i,j,k -> ",dot_1.reshape(2,2,2),q_out1_1,q_out2_1,q_out3_1) )**2#* 1/root2)**2
print('re2(|001>) = ',a2)
a3 = np.abs(np.einsum("ijk,i,j,k -> ",dot_1.reshape(2,2,2),q_out1_2,q_out2_2,q_out3_2) )**2#* 1/root2)**2
print('re3(|010>) = ',a3)
a4 = np.abs(np.einsum("ijk,i,j,k -> ",dot_1.reshape(2,2,2),q_out1_3,q_out2_3,q_out3_3) )**2#* 1/root2)**2
print('re4(|011>) = ',a4)
a5 = np.abs(np.einsum("ijk,i,j,k -> ",dot_1.reshape(2,2,2),q_out1_4,q_out2_4,q_out3_4) )**2#* 1/root2)**2
print('re5(|100>) = ',a5)
a6 = np.abs(np.einsum("ijk,i,j,k -> ",dot_1.reshape(2,2,2),q_out1_5,q_out2_5,q_out3_5) )**2#* 1/root2)**2
print('re6(|101>) = ',a6)
a7 = np.abs(np.einsum("ijk,i,j,k -> ",dot_1.reshape(2,2,2),q_out1_6,q_out2_6,q_out3_6) )**2#* 1/root2)**2
print('re7(|110>) = ',a7)
a8 = np.abs(np.einsum("ijk,i,j,k -> ",dot_1.reshape(2,2,2),q_out1_7,q_out2_7,q_out3_7) )**2#* 1/root2)**2
print('re8(|111>) = ',a8)
