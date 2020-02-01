
import numpy as np
import scipy.linalg as la
import math


"""Common Two-qubit gate"""
CNOT = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]])

SWAP = np.array([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])

"""Common Single-qubit gate"""
H = 1/math.sqrt(2) * np.array([[1, 1],
                                [1, -1]])

X = np.array([[0, 1],
            [1, 0]])

Y = np.array([[0, -1j],
            [1j, 0]], dtype = complex)

Z = np.array([[1, 0],
            [0, -1]])

"""Single-qubit rotation gate"""
def Rx(angle):
    return math.cos(angle/2) * np.eye(2) - 1j * math.sin(angle/2) * X

def Ry(angle):
    return math.cos(angle/2) * np.eye(2) - 1j * math.sin(angle/2) * Y

def Rz(angle):
    return math.cos(angle/2) * np.eye(2) - 1j * math.sin(angle/2) * Z


def Print_gate(uni_mat, qubit):
    """Print common single-qubit gate or print the matrix.

    Common single-qubit gate includes H, X, Y, Z.

    """
    if np.allclose(np.eye(2), uni_mat):
        pass
    elif np.allclose(H, uni_mat):
        print("H on ", qubit)
    elif np.allclose(X, uni_mat):
        print("X on ", qubit)
    elif np.allclose(Y, uni_mat):
        print("Y on ", qubit)
    elif np.allclose(Z, uni_mat):
        print("Z on ", qubit)
    else:
        print(uni_mat)
        print("on ", qubit)

def euler_angles_1q(unitary_matrix):
    """

    Reference, modified from qiskit package

    """
    if unitary_matrix.shape != (2, 2):
        raise QiskitError("euler_angles_1q: expected 2x2 matrix")
    phase = la.det(unitary_matrix)**(-1.0/2.0)
    U = phase * unitary_matrix  # U in SU(2)

    theta = 2 * math.atan2(abs(U[1, 0]), abs(U[0, 0]))

    # Find phi and lambda
    phiplambda = 2 * np.angle(U[1, 1])
    phimlambda = 2 * np.angle(U[1, 0])
    phi = (phiplambda + phimlambda) / 2.0
    lamb = (phiplambda - phimlambda) / 2.0

    return phase, theta, phi, lamb



def is_unitary(mat):
    """Check if the matrix is unitary

    UU* = I

    """
    return np.allclose(np.eye(len(mat)), mat.dot(mat.T.conj()))

def P1(uni_mat):
    """Decompose the two-qubit gate without CNOT gate if possible

    Decomposition without CNOT gate applies on the gate which can
    be implemented with two parallel single qubit gate(without 
    entanglement). Then, 

    U = U1(2)⊗U2(2)

    U1 = [[a, b],
        [c, d]]

    U2 = [[e, f],
        [g, h]]

    U  = uij = [[ae, af, be, bf],
                [ag, ah, bg, bh],
                [ce, cf, de, df],
                [cg, ch, dg, dh]]

    Since |det U2| = 1 and |eh - fg| = 1,

    |a| = sqrt(u00 * u11 - u01 * u10)
    a = (u00 * u11 - u01 * u10)/sqrt() or (u00 * u11 - u01 * u10)/sqrt() * i

    Same method for b, c, d.
    Then apply this factorial method on U2 but with different positions in U.
    |e| = sqrt(u00 * u22 - u02 * u20)

    """
    print("U1(2)⊗U2(2) checking...")
    U1 = uni_mat[:2, :2].copy()
    det1 = abs(U1[0, 0] * U1[1, 1] - U1[0, 1] * U1[1, 0])
    
    U2 = uni_mat[::2, ::2].copy()
    det2 = abs(U2[0, 0] * U2[1, 1] - U2[0, 1] * U2[1, 0])
    
    if det1 == 0 or det2 == 0:
        U1 = uni_mat[2:, :2].copy()
        det1 = abs(U1[0, 0] * U1[1, 1] - U1[0, 1] * U1[1, 0])

        U2 = uni_mat[::2, ::2].copy()
        det2 = abs(U2[0, 0] * U2[1, 1] - U2[0, 1] * U2[1, 0])
        if det2 == 0:
            U2 = uni_mat[1::2, ::2].copy()
            det2 = abs(U2[0, 0] * U2[1, 1] - U2[0, 1] * U2[1, 0])

    print("Tensor product decomposing...")
    U1 = U1 / np.sqrt(det1)
    U2 = U2 / np.sqrt(det2)
    if np.allclose(np.kron(U2, U1), uni_mat) and is_unitary(U1) and is_unitary(U2):
        print("Tensor product decomposing...")
        Print_gate(U1, "q1")
        Print_gate(U2, "q2")
        return True
    else:
        if np.allclose(np.kron(U2/1j, U1), uni_mat) and is_unitary(U2/1j):
            print("Tensor product decomposing...")
            U2 = U2/1j
            Print_gate(U1, "q1")
            Print_gate(U2, "q2")
            return True
        elif np.allclose(np.kron(U2, U1/1j), uni_mat) and is_unitary(U1/1j):
            print("Tensor product decomposing...")
            U1 = U1/1j
            Print_gate(U1, "q1")
            Print_gate(U2, "q2")
            return True
        else:
            print("Tensor product decomposing failed.")
            return False

def P2(uni_mat):
    """Decompose SWAP gate to 3 CNOT gates
    """
    if np.array_equal(uni_mat, SWAP):
        print("SWAP gate found.")
        print(CNOT)
        print("on q1, q2.")
        print(CNOT)
        print("on q1, q2.")
        print(CNOT)
        print("on q1, q2.")
        return True
    else:
        print("SWAP gate not found.")
        return False

def P3(uni_mat):
    """Decompose Control U gate,

    Control U = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, u1, u2],
                [0, 0, u3, u4]]

    U1 = [[u1, u2],
        [u3, u4]]

       = phase * Rz(beta) * Ry(gama) * Rz(delta)

    U = EAXBXC

    E = diag(1, phase)
    A = Rz(beta) * Ry(gama/2)
    B = Ry(-(gama)/2) * Rz(-(delta + beta)/2)
    C = Rz((delta - beta)/2)

    """
    print("Ctrl-U checking...")
    I_check = uni_mat[:2, :2]
    U = uni_mat[2:, 2:].copy()
    if np.allclose(np.eye(len(uni_mat)-2), I_check) and is_unitary(U):
        print("Ctrl-U found...")
        print("U(2) decomposing...")
        phase, beta, gama, delta = euler_angles_1q(U)
        print("U = ", phase, "* Rz(", beta, ") * Ry(", gama, ") * Rz(", delta, ")")
        print("Find A, B, C, which ABC = I and U = phase * AXBXC...")
        
        A_mat = np.dot(Rz(beta), Ry(gama/2))
        B_mat = np.dot(Ry(-(gama)/2), Rz(-(delta + beta)/2))
        C_mat = Rz((delta - beta)/2)
        E_mat = np.array([[1, 0],
                [0, phase]], dtype = complex)

        Print_gate(C_mat, "q2")
        print("CNOT on q1, q2.")
        Print_gate(B_mat, "q2")
        print("CNOT on q1, q2.")
        Print_gate(A_mat, "q2")
        Print_gate(E_mat, "q1")

        return True
    else:
        return False

#def P4

def two_qubit_decompose(uni_mat):
    print("Decompose 2-qubit gate: ")
    print(uni_mat)

    print("Unitary matrix checking...")
    if not is_unitary(uni_mat):
        print("Unitary matrix not found.")
    else:
        if not P2(uni_mat):
            if not P1(uni_mat):
                P3(uni_mat)

if __name__ == '__main__':
    ## Create an unitary_matrix SU(4)
    unitary_matrix = np.kron(H, H)
    ## Decompose it to single-qubit gate with CNOT.
    two_qubit_decompose(unitary_matrix)
    