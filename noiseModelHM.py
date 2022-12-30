import numpy as np
from qiskit import *
from qiskit import Aer
import pandas as pd
from qiskit.test.mock import *
from qiskit.providers.aer import AerSimulator, QasmSimulator
from qiskit.providers.aer.noise.noise_model import NoiseModel
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit import IBMQ
import scipy as sp
from scipy.optimize import lsq_linear
import pickle
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-L", "--L", type=int, default=1, help="System size")
    parser.add_argument("-p", "--p", type=int, default=1, help="Number of layers (currently only 1 is possible)")
    parser.add_argument("-u", "--u", type=int, default=1, help="Symmetric bounds")
    parser.add_argument("-a", "--a", type=float, default=1.47, help="Coupling strength Uf in units of Uc, 1.47 * Uc = 5")
    parser.add_argument("-t", "--t", type=int, default=5, help="Final time")
    parser.add_argument("-d", "--d", type=float, default=0.005, help="McLachlan step size")
    parser.add_argument("-s", "--s", type=int, default=2**13, help="Number of shots")
    args = parser.parse_args()
    Nl = args.L
    p = args.p
    u = args.u
    a = args.a
    dt = args.d
    tf = args.t
    shots = args.s


    def index_mu_to_lis(mu, num_layers, num_sites):
	    if mu < num_layers*(3*num_sites):
	        layer = mu // (3*num_sites)
	        i = mu % (3*num_sites)
	        if i <= 3*num_sites:
	            site = 0

	    elif mu == num_layers*(3*num_sites): # global phase term
	        layer = num_layers
	        i = 0
	        site = 0
	    else: 
	        raise (ValueError("Index mu outside bounds of ansatz. There's only one layer."))
	    return (layer, i, site)

    def circuit_M_1(m, n, thetas, num_layers, num_sites, sv_bool=True):

        if m > n : # make sure m < n (the output is symmetric)
            mm = m
            nn = n
            m = nn
            n = mm
	        
        l_m, i_m, site_m = index_mu_to_lis(m, num_layers, num_sites)
        l_n, i_n, site_n = index_mu_to_lis(n, num_layers, num_sites)
        
        qr = QuantumRegister(1, 'qb')
        ar = AncillaRegister(1, 'anc')
        cr = ClassicalRegister(1, 'cl')
        circuit = QuantumCircuit(cr,ar,qr)
        
        circuit.h(qr[0])
        circuit.h(ar[0])
        circuit.x(ar[0])
        
        # if n == global phase, only go through available layers
        if l_n == num_layers: # n = global phase parameter
            max_layer = l_n - 1
        else: 
            max_layer = l_n
        
        # go through layers 0 to l_n, unless l_n = num_layers in which case n is global phase term, and we only go through layers 0 to num_layers - 1.
        for layer in range(max_layer + 1):
            
            # if l_m = l_n
            if l_m == l_n: # could be either actual layer or global phase term (in which case l_m = l_n = num_layers)
                # build full ansatz for layers < l_m == l_n
                if layer < l_m:
                    for site in range(num_sites):
                        circuit.rz(thetas[0],qr[site])
                    for site in range(num_sites):
                        circuit.rx(thetas[1],qr[site])
                    for site in range(num_sites):
                        circuit.rz(thetas[2],qr[site])
            
                # in layer == l_m: build ansatz up to i_m, then insert controlled gate, the build ansatz for i > i_m
                elif layer == l_m: # will only be called if m is not the global phase term
                    # go through variables i < i_m
                    for i in range(i_m):
                        if i == 0: 
                            site = 0
                            circuit.rz(thetas[i],qr[site])
                        elif i == 1: 
                            site = 0
                            circuit.rx(thetas[i],qr[site])
                        else:
                            site = 0
                            circuit.rz(thetas[i],qr[site])

                    # now insert the controlled sigma_m gate (is cRzz, cRx, cRz depending on type of site i_m)
                    if i_m == 0:
                        site = 0
                        # insert cZZ
                        #print(f"insert an Rzz gate at site {site}")
                        circuit.rz(thetas[i_m], qr[site])
                        circuit.cz(ar[0],qr[0])
                    elif i_m == 1:
                        site = 0
                        # insert cX
                        circuit.rx(thetas[i_m], qr[site])
                        circuit.cx(ar[0],qr[0])
                    else:
                        site = 0
                        # insert cZ
                        circuit.rz(thetas[i_m], qr[site])
                        circuit.cz(ar[0],qr[0])
                    
                    circuit.x(ar[0])
                        
                    # go through variables i_n > i > i_m
                    for i in range(i_m+1, i_n):
                        if i == 0:
                            site = 0
                            circuit.rz(thetas[i],qr[site])
                        elif i == 1:
                            site = 0
                            circuit.rx(thetas[i],qr[site])
                        else:
                            site = 0
                            circuit.rz(thetas[i],qr[site])
                    
                    # now insert the controlled sigma_n gate (is cRzz, cRx, cRz depending on type of site i_n)
                    if i_n == 0:
                        site = 0
                        # insert cZ. Note that I do not need to insert rz here as (in contrast to i_m above).
                        circuit.cz(ar[0],qr[0])
                    elif i_n == 1:
                        site = 0
                        # insert cX. Note that I do not need to insert rx here as (in contrast to i_m above).
                        circuit.cx(ar[0],qr[0])
                    else:
                        site = 0
                        # insert cZ. Note that I do not need to insert rz here as (in contrast to i_m above).
                        circuit.cz(ar[0],qr[0])
                
                if l_m == num_layers: # m is global phase term
                    #circuit.append(cirq.ops.IdentityGate(num_sites).controlled_by(ancilla)) # control-identity gate for m
                    circuit.x(ar[0])
                    #circuit.append(cirq.ops.IdentityGate(num_sites).controlled_by(ancilla)) # control-identity gate for n
            
            else: # l_n > l_m. Therefore, m cannot be the global phase term, but n might be.
            # build full ansatz for layers < l_m
                if layer < l_m:
                    for site in range(num_sites):
                        circuit.rz(thetas[0],qr[site])
                    for site in range(num_sites):
                        circuit.rx(thetas[1],qr[site])
                    for site in range(num_sites):
                        circuit.rz(thetas[2],qr[site])

                # in layer == l_m: build ansatz up to i_m, then insert controlled gate, the build ansatz for i > i_m
                elif layer == l_m:
                    # go through variables i < i_m
                    for i in range(i_m):
                        if i == 0:
                            site = i
                            circuit.rz(thetas[0],qr[site])
                        elif i == 1:
                            site = 0
                            circuit.rx(thetas[1],qr[site])
                        else:
                            site = 0
                            circuit.rz(thetas[2],qr[site])

                    # now insert the controlled sigma_m gate (is cRzz, cRx, cRz depending on type of site i_m)
                    if i_m == 0:
                        site = 0
                        # insert cZZ
                        #print(f"insert an Rzz gate at site {site}")
                        circuit.rz(thetas[i_m], qr[site])
                        circuit.cz(ar[0],qr[0])
                    elif i_m == 1:
                        site = 0
                        # insert cX
                        circuit.rx(thetas[i_m], qr[site])
                        circuit.cx(ar[0],qr[0])
                    else:
                        site = 0
                        # insert cZ
                        circuit.rz(thetas[i_m], qr[site])
                        circuit.cz(ar[0],qr[0])

                    circuit.x(ar[0])
                    
                    # go through variables i > i_m
                    for i in range(i_m+1, 3*num_sites):
                        if i == 0:
                            site = 0
                            circuit.rz(thetas[i], qr[site])
                        elif i == 1:
                            site = 0
                            circuit.rx(thetas[i], qr[site])
                        else:
                            site = 0
                            circuit.rz(thetas[i], qr[site])

                # in layer == l_n: build ansatz up to i_n, then insert controlled gate, the build ansatz for i > i_n
                elif layer == l_n:
                    # go through variables i < i_n
                    for i in range(i_n):
                        if i == 0:
                            site = 0
                            circuit.rz(thetas[i], qr[site])
                        elif i == 1:
                            site = 0
                            circuit.rx(thetas[i], qr[site])
                        else:
                            site = 0
                            circuit.rz(thetas[i], qr[site])

                    # now insert the controlled sigma_n gate (is cRzz, cRx, cRz depending on type of site i_n)
                    if i_n == 0:
                        site = i_n
                        # insert cZZ. Note that I do not need to insert Rzz here as (in contrast to i_m above).
                        circuit.append(cirq.CZ(ancilla, qubits[site]))
                        circuit.cz(ar[0], qr[site])
                    elif i_n == 1:
                        site = 0
                        # insert cX. Note that I do not need to insert rx here as (in contrast to i_m above).
                        circuit.cx(ar[0], qr[site])
                    else:
                        site = 0
                        # insert cZ. Note that I do not need to insert rz here as (in contrast to i_m above).
                        circuit.cz(ar[0], qr[site])

        circuit.h(ar[0])
        
        if sv_bool:
            circuit.measure(ar[0],cr)

        return circuit

    def circuit_V_1(ham_type, m, thetas, num_layers, num_sites, sv_bool=True):
        
        # ham_type: 'ZZ', 'X', 'Z'
        # ham_site: (smaller) real space site that gate acts on
        ham_site = 0
        l_m, i_m, site_m = index_mu_to_lis(m, num_layers, num_sites)
        
        #print(f"l_m = {l_m}, i_m = {i_m}, site_m = {site_m}")
        
        # if n == global phase, only go through available layers
        if l_m == num_layers:
            max_layer = l_m - 1
        else: 
            max_layer = l_m
        
        qr = QuantumRegister(1, 'qb')
        ar = AncillaRegister(1, 'anc')
        cr = ClassicalRegister(1, 'cl')
        circuit = QuantumCircuit(cr,ar,qr)
        
        circuit.h(qr[0])
        circuit.h(ar[0])
        circuit.x(ar[0])
        
        # go through all layers 
        for layer in range(num_layers):
            #print(f"layer = {layer}")

            # build full ansatz for layers < l_m
            if layer < l_m:
                for site in range(num_sites):
                    circuit.rz(thetas[0], qr[site])
                for site in range(num_sites):
                    circuit.rx(thetas[1], qr[site])
                for site in range(num_sites):
                    circuit.rz(thetas[2], qr[site])

            # in layer == l_m: build ansatz up to i_m, then insert controlled gate, then build ansatz for i > i_m
            elif layer == l_m:
                # go through variables i < i_m
                for i in range(i_m):
                    if i == 0: 
                        site = 0
                        circuit.rz(thetas[i], qr[site])
                    elif i == 1: 
                        site = 0
                        circuit.rx(thetas[i], qr[site])
                    else:
                        site = 0
                        circuit.rz(thetas[i], qr[site])

                # now insert the controlled sigma_m gate (is cRzz, cRx, cRz depending on type of site i_m)
                if i_m == 0:
                    site = 0
                    circuit.rz(thetas[i_m], qr[site])
                    circuit.cz(ar[0], qr[site])
                elif i_m == 1:
                    site = 0
                    # insert cX
                    circuit.rx(thetas[i_m], qr[site])
                    circuit.cx(ar[0], qr[site])
                else:
                    site = 0
                    # insert cZ
                    circuit.rz(thetas[i_m], qr[site])
                    circuit.cz(ar[0], qr[site])

                circuit.x(ar[0])

                # go through variables i > i_m
                for i in range(i_m+1, 3*num_sites):
                    if i == 0:
                        site = 0
                        circuit.rz(thetas[i], qr[site])
                    elif i == 1:
                        site = 0
                        circuit.rx(thetas[i], qr[site])
                    else:
                        site = 0
                        circuit.rz(thetas[i], qr[site])
            
            # build full ansatz for all layers > l_m until the end
            elif (layer > l_m):
                for site in range(num_sites):
                    circuit.rz(thetas[0], qr[site])
                for site in range(num_sites):
                    circuit.rx(thetas[1], qr[site])
                for site in range(num_sites):
                    circuit.rz(thetas[2], qr[site])
            
            
        # now insert the controlled sigma_n gate that appears in the Hamiltonian (is cZZ, cRx, cRz depending on ham_type)
        if ham_type == 'Z':
            site = ham_site
            # insert cZ. Note that I do not need to insert rz here as (in contrast to i_m above).
            circuit.cz(ar[0], qr[site])
        elif ham_type == 'X':
            site = ham_site
            # insert cX. Note that I do not need to insert rx here as (in contrast to i_m above).
            circuit.cx(ar[0], qr[site])
        elif ham_type == 'I':
            site = ham_site
            # Don't do anything for the identity operator.
        else:
            raise(ValueError('ham_type not supported. Supported choices are Z, X, I.'))

        circuit.h(ar[0])
        
        if sv_bool:
            circuit.measure(ar[0],cr)
        
        return circuit
    
    def M_1(m, n, thetas, num_layers, num_sites, num_circuits, back, sv_bool=True):
        # need to append measurement gate if using run (shots)
        # circuit.append(cirq.measure(ancilla, key = 'ancilla'))
        res = execute(experiments=circuit_M_1(m, n, thetas, num_layers, num_sites, sv_bool),
                      backend=back,shots=num_circuits)
        res_ = meas_err_filter.apply(res.result())
        if res_.get_counts(0).get('1') is None:
            num = 0
        else:
            num = res_.get_counts(0)['1']
        p1 = num/sum(res_.get_counts(0).values())
        M_1_res = 1 - 2.*p1
        return M_1_res

    def emm():
        M_1_matrix = np.zeros((len(thetas), len(thetas)))
        for m in range(len(thetas)):
            #print(f'm = {m}')
            for n in range(m, len(thetas)):
                #print(f'n = {n}')
                M_1_matrix[m, n] = 2*M_1(m, n, thetas, p, Nl, shots, back)
                if n > m:
                    M_1_matrix[n, m] = M_1_matrix[m, n]
        return M_1_matrix

    def vee(n):
        
        V_vector = np.zeros((len(thetas)))

        Uc = 32/(3*np.pi)
        Uf = n*Uc

        qr = QuantumRegister(1, 'qb')
        cr = ClassicalRegister(1, 'cl')
        sq = QuantumCircuit(cr,qr)
        
        sq.h(qr[0])
        sq.rz(thetas[0], qr[0])
        sq.rx(thetas[1], qr[0])
        sq.rz(thetas[2], qr[0])
        sq.h(qr[0])
        sq.measure(qr[0], cr)
        res_sq = execute(experiments=sq, backend=back, shots=shots)
        res_fin = meas_err_filter.apply(res_sq.result())
        if res_fin.get_counts(0).get('1') is None:
            num = 0
        else:
            num = res_fin.get_counts(0)['1']
        p2 = num/sum(res_fin.get_counts(0).values())

        Delta = np.real(1 - 2*p2)/4

        hiVal = Uc/4
        hxVal = -Uc*Delta
        hzVal = -Uf/4

        for m in range(len(thetas)):

            result_V1 = 0.

            for ham_type in ['I', 'X', 'Z']:
                for ham_site in range(Nl):
                    res = execute(experiments=circuit_V_1(ham_type, m ,thetas, p, Nl),
                                 backend=back, shots=shots)
                    res_ = meas_err_filter.apply(res.result())
                    if res_.get_counts(0).get('1') is None:
                        num = 0
                    else:
                        num = res_.get_counts(0)['1']
                    p1 = num/sum(res_.get_counts(0).values())
                    Pauli_ancilla = 1 - 2*p1
                    if ham_type == 'Z':
                        result_V1 += hzVal*Pauli_ancilla
                        #print(f"hz*Z[{ham_site}] = {hzVal*Pauli_ancilla}")
                    elif ham_type == 'X':
                        result_V1 += hxVal*Pauli_ancilla
                        #print(f"hx*X[{ham_site}] = {hxVal*Pauli_ancilla}")
                    elif ham_type == 'I':
                        result_V1 += hiVal*Pauli_ancilla
                        #print(f"hx*X[{ham_site}] = {hxVal*Pauli_ancilla}")

            result_V = 2.*(result_V1) # + np.imag(np.conj(result_M2)*result_H))
            V_vector[m] = np.real(result_V)

        return V_vector

    def doc():
        
        qr = QuantumRegister(1, 'qb')
        cr = ClassicalRegister(1, 'cl')
        sz = QuantumCircuit(cr,qr)
        
        sz.h(qr[0])
        sz.rz(thetas[0], qr[0])
        sz.rx(thetas[1], qr[0])
        sz.rz(thetas[2], qr[0])
        sz.measure(qr[0], cr)
        res_sz = execute(experiments=sz, backend=back, shots=shots)
        res_fin = meas_err_filter.apply(res_sz.result())
        if res_fin.get_counts(0).get('1') is None:
            num = 0
        else:
            num = res_fin.get_counts(0)['1']
        p2 = num/sum(res_fin.get_counts(0).values())

        zee = np.real(1 - 2*p2)

        zee = np.real(1 - 2*p2)
        return (1 - zee)/4


    device_backend = FakeSantiago()
    # back = Aer.get_backend('aer_simulator')
    back = AerSimulator.from_backend(device_backend)

    qr = QuantumRegister(Nl)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    t_qc = transpile(meas_calibs, back) # , initial_layout=qb_map
    qobj = assemble(t_qc, shots=32000)
    cal_results = back.run(qobj, shots=32000).result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    meas_err_filter = meas_fitter.filter

    b = (-u,u)

    thetas = np.array([0.,0.,0.,0.])

    delt_thet = [thetas.copy()]

    doub_occ = [doc()]

    for _ in range(int(tf/dt)):
        M = emm()
        V = vee(a)
        dtheta = (lsq_linear(M,V,b).x)*2*dt
        # dtheta = np.real(np.array(np.linalg.inv(M).dot(V) * dtt))
        thetas += dtheta
        delt_thet.append(thetas.copy())
        
        doub_occ.append(doc())
    
    if (os.path.exists(f'./wiNoise/mNoise_{dt}_{shots}_{u}_{tf}/')):
        with open(f'./wiNoise/mNoise_{dt}_{shots}_{u}_{tf}/thetas_{dt}_{shots}_{u}_{tf}.pkl','wb') as f:
            pickle.dump(delt_thet,f)

        with open(f'./wiNoise/mNoise_{dt}_{shots}_{u}_{tf}/doub_occ_{dt}_{shots}_{u}_{tf}.pkl','wb') as f:
            pickle.dump(doub_occ,f)
    else:
        os.mkdir(f'./wiNoise/mNoise_{dt}_{shots}_{u}_{tf}')
        with open(f'./wiNoise/mNoise_{dt}_{shots}_{u}_{tf}/thetas_{dt}_{shots}_{u}_{tf}.pkl','wb') as f:
            pickle.dump(delt_thet,f)

        with open(f'./wiNoise/mNoise_{dt}_{shots}_{u}_{tf}/doub_occ_{dt}_{shots}_{u}_{tf}.pkl','wb') as f:
            pickle.dump(doub_occ,f)











