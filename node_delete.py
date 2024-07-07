
#ノードの消去
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

#量子ゲート
root2 = ma.sqrt(2)
h1 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#1
h2 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#2
h3 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#3
h4 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#1
h5 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#2
h6 = np.array([[1/root2,1/root2],[1/root2,-1/root2]])#3

x_gate = np.array([[0,1],
                   [1,0]])

CZ_13 = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,-1]]).reshape(2,2,2,2)

CZ_12 = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,-1]]).reshape(2,2,2,2)

CZ_23 = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,-1]]).reshape(2,2,2,2)

#i14を消去
i14_delete = np.einsum('ij,jk -> ik',x_gate,h4)

#i11を消去
i11_delete = np.einsum('ij,jklm,lnop,oq -> inkqpm',h1,CZ_13,CZ_12,i14_delete)

#i33を消去
i33_delete = np.einsum('ij,kjlm,mn -> kiln',x_gate,CZ_23,h6)

#i21を消去
i21_delete = np.einsum('ij,kjlmno,nopq,pr -> kilmrq',h2,i11_delete,i33_delete,h5)

#i31を消去
i31_delete = np.einsum('ij , kljmno -> klimno',h3,i21_delete)

#出力状態ごとに縮約
re1 = np.einsum('i,j,k,ijklmn,l,m,n -> ',q_in1,q_in2,q_in3,i31_delete,q_out1_0,q_out2_0,q_out3_0)
re2 = np.einsum('i,j,k,ijklmn,l,m,n -> ',q_in1,q_in2,q_in3,i31_delete,q_out1_1,q_out2_1,q_out3_1)
re3 = np.einsum('i,j,k,ijklmn,l,m,n -> ',q_in1,q_in2,q_in3,i31_delete,q_out1_2,q_out2_2,q_out3_2)
re4 = np.einsum('i,j,k,ijklmn,l,m,n -> ',q_in1,q_in2,q_in3,i31_delete,q_out1_3,q_out2_3,q_out3_3)
re5 = np.einsum('i,j,k,ijklmn,l,m,n -> ',q_in1,q_in2,q_in3,i31_delete,q_out1_4,q_out2_4,q_out3_4)
re6 = np.einsum('i,j,k,ijklmn,l,m,n -> ',q_in1,q_in2,q_in3,i31_delete,q_out1_5,q_out2_5,q_out3_5)
re7 = np.einsum('i,j,k,ijklmn,l,m,n -> ',q_in1,q_in2,q_in3,i31_delete,q_out1_6,q_out2_6,q_out3_6)
re8 = np.einsum('i,j,k,ijklmn,l,m,n -> ',q_in1,q_in2,q_in3,i31_delete,q_out1_7,q_out2_7,q_out3_7)
print("input -> |000>")
print('re1(|000>) = ', np.abs(re1)**2)
print('re2(|001>) = ', np.abs(re2)**2)
print('re3(|010>) = ', np.abs(re3)**2)
print('re4(|011>) = ', np.abs(re4)**2)
print('re5(|100>) = ', np.abs(re5)**2)
print('re6(|101>) = ', np.abs(re6)**2)
print('re7(|110>) = ', np.abs(re7)**2)
print('re8(|111>) = ', np.abs(re8)**2)

#print(np.einsum('i,j,k,ijklmn -> lmn',q_in1,q_in2,q_in3,c_delete))




print('total;',np.abs(re1)**2 + np.abs(re2)**2 + np.abs(re3)**2 + np.abs(re4)**2 + np.abs(re5)**2 + np.abs(re6)**2 + np.abs(re7)**2 +np.abs(re8)**2)

