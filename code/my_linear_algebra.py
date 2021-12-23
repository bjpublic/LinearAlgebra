#!/usr/bin/env python
# coding: utf-8

# In[2]:


def zero_mat(n, p):
    """
    영 행렬 생성
    입력값: 생성하고자 할 영 행렬의 크기 n행, p열
    출력값: nxp 영 행렬 Z
    """
    Z = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            row.append(0)
        Z.append(row)
    return Z



def deepcopy(A):
    """
    깊은 복사(deepcopy) 구현
    입력값: 깊은 복사를 하고자 하는 행렬 리스트 A
    출력값: 깊은 복사 된 결과 행렬 리스트 res 
    """
    if type(A[0]) == list:
        n = len(A)
        p = len(A[0])
        res = zero_mat(n,p)
        for i in range(0,n):
            for j in range(0,p):
                res[i][j] = A[i][j]
        return res
    else:
        n = len(A)
        res = []
        for i in range(0,n):
            res.append(A[i])
        return res

    
    
def v_add(u,v):
    """
    벡터의 덧셈
    입력값: 더하고자 하는 벡터 u, v
    출력값: 벡터 u, v의 덧셈 결과 w
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = u[i] + v[i]
        w.append(val) 
        
    return w


def v_subtract(u,v):
    '''
    벡터의 뺄셈
    입력값: 빼고자하는 벡터 리스트 u,v
    출력값: 벡터 u,v의 뺄셈 결과 w
    '''
    n = len(u)
    w = []

    for i in range(0, n):
        val = u[i] - v[i]
        w.append(val)
    return w


def scalar_v_mul(a,u):
    """
    벡터의 스칼라 곱
    입력값: 스칼라 a, 벡터 리스트 u
    출력값: 스칼라 a와 벡터 리스트 u의 곱 결과 w
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = a*u[i]
        w.append(val) 
        
    return w


def v_mul(u,v):
    """
    벡터의 원소 곱
    입력값: 원소 곱 하고자할 벡터 리스트 u, v
    출력값: 벡터 u, v의 원소 곱 결과 w
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = u[i] * v[i]
        w.append(val) 
    
    return w



def v_div(u,v):
    """
    벡터의 원소 나눗셈
    입력값: 원소 나눗셈 하고자할 벡터 리스트 u, v
    출력값: 벡터 u, v의 원소 나눗셈 결과 w    
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = u[i] / v[i]
        w.append(val) 
    
    return w


def add(A, B):
    """
    행렬의 덧셈
    입력값: 행렬의 덧셈을 수행할 행렬 A, B
    출력값: 행렬 A와 행렬 B의 덧셈 결과인 행렬 res
    """

    n = len(A)
    p = len(A[0])
    
    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[i][j] + B[i][j]
            row.append(val)
        res.append(row)
    return res


def subtract(A, B):
    """
    행렬의 뺄셈
    입력값: 행렬의 뺄셈을 수행할 행렬 A, B
    출력값: 행렬 A와 행렬 B의 뺄셈 결과인 행렬 res
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[i][j] - B[i][j]
            row.append(val)
        res.append(row)
    return res


def scalar_mul(b, A):
    """
    행렬의 스칼라곱
    입력값: 스칼라곱을 수행할 스칼라 b, 행렬 A
    출력값: 스칼라 b와 행렬 A의 스칼라 곱 결과인 행렬 res
    """

    n = len(A)
    p = len(A[0])
    
    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = b * A[i][j]
            row.append(val)
        res.append(row)
    return res



def ele_product(A,B):
    """
    행렬의 원소곱
    입력값: 행렬의 원소곱을 수행할 행렬 A, B
    출력값: 행렬 A와 행렬 B의 원소곱 결과인 행렬 res
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[i][j] * B[i][j]
            row.append(val)
        res.append(row)
    return res



def matmul(A, B):    
    """
    행렬의 행렬곱
    입력값: 행렬곱을 수행할 행렬 A, B
    출력값: 행렬 A와 행렬 B의 행렬곱 결과인 행렬 res
    """
    n = len(A)
    p1 = len(A[0])
    p2 = len(B[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p2):
            val = 0
            for k in range(0, p1):
                val += A[i][k] * B[k][j] 
            row.append(val)    
        res.append(row)
    return res



def transpose(A):
    """
    행렬의 전치행렬
    입력값: 전치행렬을 구하고자 하는 행렬 A
    출력값: 행렬 A의 전치행렬 At
    """
    n = len(A)
    p = len(A[0])

    At  = []
    for i in range(0, p):
        row = []
        for j in range(0, n):
            val = A[j][i]
            row.append(val)
        At.append(row)
    return At


def diag(A):
    """
    행렬의 대각행렬
    입력값: 대각행렬을 구하고자 하는 행렬 A
    출력값: 행렬 A의 대각행렬 D
    """
    n = len(A)
    D = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i==j:
                row.append(A[i][j])
            else:
                row.append(0)
        D.append(row)
    return D



def diag_ele(A):
    """
    대각 원소 구하기
    입력값: 대각 원소를 구하고자 할 행렬 A
    출력값: 행렬 A의 대각 원소 리스트 d
    """
    n = len(A)
    d = []
    for i in range(0, n):
        d.append(A[i][i])
    return d


def ele2diag(a):
    """
    대각원소 -> 대각행렬 변환
    입력값: 대각 원소 리스트 a
    출력값: 대각 원소 a를 이용해 생성한 nxn 대각 행렬 D
    n: 대각 원소 리스트 a의 길이
    """
    n = len(a)
    D = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i==j:
                row.append(a[i])
            else:
                row.append(0)
        D.append(row)
    return D


def identity(n):
    """
    항등행렬 생성
    입력값: 항등 행렬의 크기 n
    출력값: nxn 항등 행렬 I
    """
    I  = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i==j:
                row.append(1)
            else:
                row.append(0)
        I.append(row)
    return I


def u_tri(A):
    """
    상 삼각 행렬 변환
    입력값: 상 삼각 행렬로 변환하고자 하는 행렬 A
    출력값: 행렬 A를 상 삼각 행렬로 변환시킨 행렬 utri
    """
    n = len(A)
    p = len(A[0])
    utri  = []

    for i in range(0, n):
        row = []
        for j in range(0, p):
            if i>j:
                row.append(0)
            else:
                row.append(A[i][j])
        utri.append(row)
    return utri


def l_tri(A):
    """
    하 삼각 행렬 변환
    입력값: 하 삼각 행렬로 변환하고자 하는 행렬 A
    출력값: 행렬 A를 하 삼각 행렬로 변환시킨 행렬 ltri
    """
    n = len(A)
    p = len(A[0])
    ltri  = []

    for i in range(0, n):
        row = []
        for j in range(0, p):
            if i<j:
                row.append(0)
            else:
                row.append(A[i][j])
        ltri.append(row)
    return ltri


def toeplitz(a, b):
    """
    토플리츠 행렬 변환
    입력값: 토플리츠 행렬로 변환하고자 하는 리스트 a, b
    출력값: 리스트 a, b를 이용해 만든 토플리츠 행렬 A
    """
    n1 = len(a)
    n2 = len(b)
    A = []
    for i in range(0, n1):
        row = []
        for j in range(0, n2):
            if i > j:
                row.append(a[i-j])
            else:
                row.append(b[j-i])
        A.append(row)
    return A


def u_bidiag(A):
    """
    upper bidiagonal 행렬
    입력값: 행렬 A
    출력값: 행렬 A의 upper bidiagonal 행렬 res
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0,n):
        row = []
        for j in range(0,p):
            if i > j or j-i > 1:
                row.append(0)
            else:
                row.append(A[i][j])
        res.append(row) 
    return res


def l_bidiag(A):
    """
    lower bidiagonal 행렬
    입력값: 행렬 A
    출력값: 행렬 A의 lower bidiagonal 행렬 res
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0,n):
        row = []
        for j in range(0,p):
            if i < j or i-j > 1:
                row.append(0)
            else:
                row.append(A[i][j])
        res.append(row)   
    return res


def inner_product(a, b):
    """
    벡터의 내적
    입력값: 내적할 벡터 리스트 a, b
    출력값: 벡터 a, b의 내적 결과 res
    """
    n = len(a)
    res = 0
    for i in range(0, n):
        res += a[i]*b[i]
    return res



def outer_product(a, b):
    """
    벡터의 외적
    입력값: 외적할 벡터 리스트 a, b
    출력값: 벡터 a, b의 외적 결과 res
    """
    res = []
    n1 = len(a)
    n2 = len(b)
    for i in range(0, n1):
        row = []
        for j in range(0, n2):
            val = a[i]*b[j]
            row.append(val)
        res.append(row)
    return res


def householder(v):
    """
    하우스홀더 행렬
    입력값: 하우스홀더 행렬을 생성할 리스트 v
    출력값: 리스트 v를 이용해 생성한 하우스홀더 행렬 H
    """
    n = len(v)
    outer_mat = outer_product(v,v)
    inner_val = inner_product(v,v)
    V = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            val = (2/inner_val)*outer_mat[i][j]
            row.append(val)
        V.append(row)
    H = subtract(identity(n), V)
    return H


def solve(A, b):
    """
    연립 방정식 풀기
    입력값: 솔루션을 구하고 싶은 A, b
    출력값: 방정식의 솔루션 sol
    """
    
    X = deepcopy(A)
    sol = deepcopy(b)
    n = len(X)

    for i in range(0,n):
        x_row = X[i]
        y_val = sol[i]

        if x_row[i]!=0:
            tmp = 1/x_row[i]
        else:
            tmp = 0

        x_row = [element * tmp for element in x_row]
        y_val = y_val*tmp

        X[i] = x_row
        sol[i] = y_val

        for j in range(0,n):
            if i==j:
                continue
            x_next = X[j]
            y_next = sol[j]
            x_tmp = [element * -x_next[i] for element in x_row]
            y_tmp = y_val*(-x_next[i])

            for k in range(0, len(x_row)):
                x_next[k] = x_tmp[k] + x_next[k]
            y_next = y_tmp + y_next

            X[j] = x_next
            sol[j] = y_next

    return sol



def norm(a):
    """
    벡터의 norm
    입력값: norm을 구하고자 할 벡터 a
    출력값: 벡터 a의 norm 결과 res
    """
    n = len(a)
    res = 0
    for i in range(0, n):
        res += a[i]**2
    res = res**(0.5)   
    return res



def normalize(a):
    """
    벡터 a의 normalization
    벡터 a의 norm을 1로 만들어줌
    입력값: normalization할 벡터 리스트 a
    출력값: 벡터 a를 normalization한 결과 벡터 리스트 v
    """
    n = len(a)
    v = []
    for i in range(0,n):
        tmp = a[i]/norm(a)
        v.append(tmp)
    return v



def det_rec(A):
    
    """
    행렬 A의 행렬식 구하기(재귀 방식을 이용)
    입력값: 행렬식을 구하고자 하는 행렬 A
    출력값: 행렬 A의 행렬식 res
    """
        
    n = len(A)
    res = 0

    # 2x2 행렬의 행렬식 구하기
    if n == 2:
        res = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return res

    # nxn 행렬의 행렬식 구하기
    for i in range(0,n):  
        X = deepcopy(A)  
        X = X[1:]  
        nx = len(X)

        for j in range(0,nx):  
            X[j] = X[j][0:i] + X[j][i+1:]  

        sign = (-1) ** (i % 2)  
        sub_det = det_rec(X)  
        res += sign * A[0][i] * sub_det 

    return res


def det_tri(A):
    """
    상 삼각 행렬 변환을 이용해 행렬식 구하기
    입력값: 행렬 A
    출력값: 행렬식 res
    """
    n = len(A)
    X = deepcopy(A)
    n_row_change = 0
 
    for i in range(0,n): 
        if X[i][i] == 0:
            tmp = X[i+1]
            X[i+1] = X[i]
            X[i] = tmp
            n_row_change += 1
        for j in range(i+1,n):
            ratio = X[j][i] / X[i][i] 
            for k in range(0, n):
                X[j][k] = X[j][k] - ratio * X[i][k]
    
    n_row_change = (-1)**(n_row_change)    
    res = 1
    for i in range(n):
        res *= X[i][i] 
    res *= n_row_change  
    return res


def inv(A):
    """
    행렬 A의 역행렬 구하기
    입력값: 행렬 A
    출력값: 행렬 A의 역행렬 res
    """
    
    n = len(A)
    X = deepcopy(A)
    
    C = []
    for i in range(0, n):
        row_C = []
        idx_r = list(range(0, n))
        idx_r.remove(i)
        for j in range(0, n):
            idx_c = list(range(0,n))
            idx_c.remove(j)
            M = []
            for k in idx_r:
                row_M = []
                for l in idx_c:
                    val = X[k][l]
                    row_M.append(val)
                M.append(row_M)
            Mij = det_rec(M)
            Cij = ((-1)**(i+j))*Mij
            row_C.append(Cij)
        C.append(row_C)
    
    adj = transpose(C)
    res = scalar_mul(1/det_rec(X), adj)
    
    return res



def qr_gram(A):
    """
    그램 슈미트 방법을 이용한 QR분해
    입력값: 행렬 A
    출력값: 행렬 A를 그램 슈미트 방법을 이용해 QR분해한 결과 행렬 Q, R
    """
    n = len(A)
    p = len(A[0])

    At = transpose(A)

    U = []
    norm_list = []

    V = []
    Q = []
    R = []

    for i in range(0,n):
        if i == 0:
            u = At[i]
            norm_u = norm(u)
            U.append(u)
            norm_list.append(norm_u)
        else:
            a = At[i]
            dp_list = []
            for j in range(0,i):
                dp = inner_product(a, U[j])
                dp_list.append(dp)

            u = []
            for j in range(0,n):
                val = a[j]
                for k in range(0,i):
                    val -= (dp_list[k]/norm_list[k]**2)*U[k][j]
                u.append(val)
            norm_u = norm(u)
            U.append(u)
            norm_list.append(norm_u)

        v = normalize(u)
        V.append(v)        

    Q = transpose(V)

    for i in range(0,n):
        r = []
        for j in range(0,n):
            if i > j:
                r.append(0)
            else:
                r_ele = inner_product(At[j], V[i])
                r.append(r_ele)
        R.append(r)
    
    return Q, R




def sign(a):
    """
    스칼라 a의 부호
    입력값: 스칼라 a
    출력값: 스칼라 a가 a>=0면 1 출력, a<0이면 0 출력
    """
    res = 1
    if a < 0:
        res = -1
    return res



def eig_qr(A):
    """
    이 방법은 행렬 A가 대칭행렬이여야만 사용할수있음
    QR분해를 이용한 고윳값, 고유벡터 구하기
    인풋: 고윳값, 고유벡터를 구하고자 하는 행렬 A
    아웃풋: E = 고윳값, V = 고유벡터
    """
    n = len(A)
    E = deepcopy(A)
    V = identity(n)
    for i in range(0, 30):
        Q, R = qr_gram(E)
        E = matmul(R, Q)
        V = matmul(V, Q)
    
    return E, V





    
def qr_householder(A):
    """
    행렬 A의 하우스홀더 방법을 이용한 QR분해
    입력값: 행렬 A
    출력값: 행렬 Q, 행렬 R
    """
        
    n = len(A)
    p = len(A[0])
    
    H_list = []
    
    for i in range(0, p):
        
        if i==0:
            A1 = deepcopy(A)
            exA = A1
        elif i < p-1:
            Ai = []
            for j in range(1, len(exA)):
                row = []
                for k in range(1, len(exA[0])):
                    row.append(HA[j][k])
                Ai.append(row)
            exA = Ai
        elif i==p-1:
            Ap = []
            for j in range(1, len(HA)):
                Ap.append(HA[j][1])
            exA = Ap
                
        # 열 추출
        if i < p-1:
            a = transpose(exA)[0]
        else:
            a = exA
        nm = norm(a)
        
        # e 생성
        e = [1]
        for j in range(0, len(a)-1):
            e.append(0)
        
        # v 생성
        tmp_e = []
        for j in range(0, len(e)):
            val = sign(a[0])*nm*e[j]
            tmp_e.append(val)
        v = v_add(a, tmp_e)    
        
        # H 생성
        H = householder(v)    
            
        # H*A
        if i==p-1:
            HA = []
            for j in range(0,len(H)):
                val = 0
                for k in range(0, len(H[0])):
                    val += H[j][k]*exA[k]
                HA.append(val)
        else:
            HA = matmul(H, exA)
        
        H_list.append(H)
            
        if i > 0:
            tmp_H = identity(len(A))
            for j in range(i, len(A)):
                for k in range(i, len(A)):
                    tmp_H[j][k] = H_list[-1][j-i][k-i]
            H_list[-1] = tmp_H

    Q = deepcopy(H_list[0])
    for j in range(0, len(H_list)-1):
        Q = matmul(Q, H_list[j+1])

    R = deepcopy(H_list[-1])
    for j in range(1,len(H_list)):
        R = matmul(R, H_list[-(j+1)])
    R = matmul(R, A)
        
    return Q, R


def svd(A):
    """
    svd를 이용한 고윳값, 고유벡터 구하기
    입력값: 고윳값, 고유벡터를 구하고자 하는 행렬 A
    출력값: U = 고유벡터, S = 특이값, Vt = AtA의 고유벡터
    """
    At = transpose(A)
    AtA = matmul(At, A)
    E, V = eig_qr(AtA)
    n = len(AtA)
    for i in range(0, n):
        E[i][i] = E[i][i]**0.5
    S = diag(E)
    Vt = transpose(V)
    
    AV = matmul(A, V)
    AVt = transpose(AV)
    Ut = []
    for vector in AVt:
        Ut.append(normalize(vector))
    U = transpose(Ut)
    return U, S, Vt


def lu_decomp(A):
    """
    LU 분해
    입력값: 행렬 A
    출력값: 행렬 A의 LU분해 행렬 L, U
    """
    n = len(A)
    p = len(A[0])

    L = [[0]*p for i in range(0, n)]
    U = []

    for i in range(0, n):
        a = A[i]
        val = 1/a[i]
        L[i][i] = 1/val
        a = [element * val for element in a] 
        U.append(a)

        for j in range(i+1, n):
            row = A[j]
            a_tmp = [element * -row[i] for element in a]
            L[j][i] = row[i]        
            A[j] = [a_tmp[k] + row[k] for k in range(p)]

    return L, U


# In[ ]:




