#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qiskit import *
from qiskit import Aer
import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator, QasmSimulator
from qiskit import IBMQ
from timeit import default_timer as timer
import scipy as sp
from scipy import sparse
from scipy.optimize import lsq_linear
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-a", "--a", type=float, default=1.47, help="Coupling strength Uf in units of Uc, 1.47 * Uc = 5")
    parser.add_argument("-b", "--b", type=int, default=2, help="Symmetric bounds for bounded minimization")
    parser.add_argument("-t", "--t", type=int, default=5, help="Final time")
    parser.add_argument("-d", "--d", type=float, default=0.001, help="McLachlan step size")
    parser.add_argument("-s", "--s", type=int, default=2**13, help="Number of shots")
    args = parser.parse_args()
    b = args.b
    a = args.a
    dt = args.d
    tf = args.t
    shots = args.s

    # The functions

    def index_mu_to_lis(mu, num_layers, num_sites):

        ''' This exists to make generalizing to more than one layer eventually easier. '''

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

    def circuit_M_1_param(m, n, num_layers, num_sites):

        ''' This builds the parameterized M circuits. '''

        if m > n : # make sure m < n (the output is symmetric)
            mm = m
            nn = n
            m = nn
            n = mm
            
        l_m, i_m, site_m = index_mu_to_lis(m, num_layers, num_sites)
        l_n, i_n, site_n = index_mu_to_lis(n, num_layers, num_sites)
        
        # There parameters, tee, are external to this function: theta is used internally.
        thetas = tee
        
        # Instantiate the circuit elements and set the initial state (1/sqrt(2)(|0> + |1>)) 
        qr = QuantumRegister(1, 'qb')
        ar = QuantumRegister(1, 'anc')
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
        
        # go through layers 0 to l_n, unless l_n = num_layers in which case n is global phase term, 
        # and we only go through layers 0 to num_layers - 1. For now, there is only 1 layer.
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

                    # now insert the controlled sigma_m gate (is cz or cx)
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
                    
                    # now insert the controlled sigma_n gate (is cz or cx)
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
                    circuit.x(ar[0])
            
            else: # l_n > l_m. Therefore, m cannot be the global phase term, but n might be.
            # build full ansatz for layers < l_m 
            ### Note: This never activates for now because there's only ever 1 layer ###
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

                    # now insert the controlled sigma_m gate (is cz or cx)
                    if i_m == 0:
                        site = 0
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

                    # now insert the controlled sigma_n gate (is cz or cx)
                    if i_n == 0:
                        site = i_n
                        # insert cz. Note that I do not need to insert rz here as (in contrast to i_m above).
                        circuit.cz(ar[0], qr[site])
                    elif i_n == 1:
                        site = 0
                        # insert cx. Note that I do not need to insert rx here as (in contrast to i_m above).
                        circuit.cx(ar[0], qr[site])
                    else:
                        site = 0
                        # insert cz. Note that I do not need to insert rz here as (in contrast to i_m above).
                        circuit.cz(ar[0], qr[site])
            
        circuit.h(ar[0])
        
        circuit.measure(ar[0],cr)

        return circuit
        
    def circuit_V_1_param(ham_type, m, num_layers, num_sites):

        ''' This builds the parameterized V circuits. '''
        
        # ham_type: 'Z', 'X', 'I'
        # ham_site: (smaller) real space site that gate acts on
        ham_site = 0
        l_m, i_m, site_m = index_mu_to_lis(m, num_layers, num_sites)
        
        # Parameters tee are external; theta is the internal variables.
        thetas = tee
        
        # if n == global phase, only go through available layers (again, only 1 layer for now)
        if l_m == num_layers:
            max_layer = l_m - 1
        else: 
            max_layer = l_m
        
        # Make the initial circuit and state: (1/sqrt(2)(|0> + |1>))
        qr = QuantumRegister(1, 'qb')
        ar = QuantumRegister(1, 'anc')
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

                # now insert the controlled sigma_m gate (is cz or cx)
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
            
            
        # now insert the controlled sigma_n gate that appears in the Hamiltonian (is cz or cx depending on ham_type)
        if ham_type == 'Z':
            site = ham_site
            # insert cz. Note that I do not need to insert rz here as (in contrast to i_m above).
            circuit.cz(ar[0], qr[site])
        elif ham_type == 'X':
            site = ham_site
            # insert cx. Note that I do not need to insert rx here as (in contrast to i_m above).
            circuit.cx(ar[0], qr[site])
        elif ham_type == 'I':
            site = ham_site
            # Don't do anything for the identity operator.
        else:
            raise(ValueError('ham_type not supported. Supported choices are Z, X, I.'))

        circuit.h(ar[0])
        
        circuit.measure(ar[0],cr)
        
        return circuit


    def circuit_X_param():

        ''' Parameterized circuit to measure the expectation value of X, i.e., to find Delta. 
        Note that the global phase can be ignored since the identity commutes with everything. '''
        
        thetas = tee
        
        qr = QuantumRegister(1, 'qb')
        cr = ClassicalRegister(1, 'cl')
        sq = QuantumCircuit(cr,qr)
        
        sq.h(qr[0])
        sq.rz(thetas[0], qr[0])
        sq.rx(thetas[1], qr[0])
        sq.rz(thetas[2], qr[0])
        sq.h(qr[0])
        sq.measure(qr[0], cr)
        
        return sq

    def circuit_do_param():

        ''' Parameterized circuit to find the double occupancy, D(t). Note that the
        global phase term can be ignored since the identity commutes with everything. '''
        
        thetas = tee
        
        qr = QuantumRegister(1, 'qb')
        cr = ClassicalRegister(1, 'cl')
        sz = QuantumCircuit(cr,qr)
        
        sz.h(qr[0])
        sz.rz(thetas[0], qr[0])
        sz.rx(thetas[1], qr[0])
        sz.rz(thetas[2], qr[0])
        sz.measure(qr[0], cr)
        
        return sz

    # The script

    # The try statement restarts the script when the Runtime API kicks me out.
    try:

        print('Program started.')

        Uc = 32/(3*np.pi)
        Uf = a*Uc

        # To use this you must have previously run:
        # QiskitRuntimeService.save_account(channel="ibm_quantum", token="MY_IBM_QUANTUM_TOKEN")
        # in order for IBM to accept your request. Find your quantum token in your IBMQ account.
        service = QiskitRuntimeService()

        # Change this between the two in order to run on either the qasm simulator or a real device.
        # options = {'backend_name': 'ibmq_qasm_simulator'}
        options = {'backend_name': 'ibmq_quito'}
        resilience = {'level': 1}
        mt = '2h 40m 00s'

        # The parameters.
        tee = ParameterVector('p',4)

        # Build a list of all the necessary circuits in order to use the 'with' command below.
        all_circuits = []
        m_loc = []
        v_loc = []

        # May want to exclude the diagonal because it is always 2. Set range(i,len(tee)) to range(i+1,len(tee)) instead.
        for i in range(len(tee)):
            for j in range(i,len(tee)):
                all_circuits.append(circuit_M_1_param(i, j, 1, 1))
                m_loc.append((len(all_circuits)-1, i, j))
                
        for op in ['Z', 'X', 'I']:
            for i in range(len(tee)):
                all_circuits.append(circuit_V_1_param(op, i, 1, 1))
                v_loc.append((len(all_circuits)-1, op, i))
                
        all_circuits.append(circuit_X_param())

        all_circuits.append(circuit_do_param())

        # Make the list of thetas to accept updates.
        t0, t1, t2, t3 = 0., 0., 0., 0.
        thetas = [t0, t1, t2, t3]

        # , resilience_settings=resilience, max_time = mt <-- this could go into the Sampler() call.
                
        with Sampler(circuits=all_circuits, service=service, options=options) as sampler:
            
            # Note that parameters never appear out of order in a circuit. That is, there's never a case
            # where, e.g., you have p0 and p2 but not p1. If p2 exists in a circuit, then so does p1. That
            # fact goes into how I assign parameters to a circuit (Runtime won't intelligently assign
            # parameters to a circuit out of a body of available parameters -- you have to tell it which
            # parameters go into each circuit explicitly).
            param_do_init = thetas[0:len(all_circuits[-1].parameters)]
            
            # This script will make a new folder with new .csv files if they don't exist already. If they
            # do exist already, it will try to scrape the last set of data out of the .csv files and use
            # that to perform the next time step.
            if os.path.exists(f'./resQT_{shots}_{a}_{dt}_{tf}/'):
                if os.path.exists(f'./resQT_{shots}_{a}_{dt}_{tf}/th_{shots}_{a}_{dt}_{tf}.csv'):
                    thets = pd.read_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/th_{shots}_{a}_{dt}_{tf}.csv', index_col=0)
                    thetas = list(thets.iloc[-1])
                    doub_occ = pd.read_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/do_{shots}_{a}_{dt}_{tf}.csv', index_col=0)
                    ts = thets.index
                else:
                    res_ = sampler(circuits=[len(all_circuits)-1], 
                               parameter_values=[param_do_init],
                              shots=shots)
                    if res_.quasi_dists[0].get('1') == None:
                        inter = 0.
                    else:
                        inter = res_.quasi_dists[0].get('1')
                    thets = pd.DataFrame(np.array([thetas]), index=np.array([0]))
                    thets.to_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/th_{shots}_{a}_{dt}_{tf}.csv')
                    doub_occ = pd.DataFrame(np.array([(1 - (1 - 2 * inter))/4]), index=np.array([0]))
                    doub_occ.to_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/do_{shots}_{a}_{dt}_{tf}.csv')
                    ts = thets.index
            else:
                os.mkdir(f'./resQT_{shots}_{a}_{dt}_{tf}/')
                res_ = sampler(circuits=[len(all_circuits)-1], 
                           parameter_values=[param_do_init],
                          shots=shots)
                if res_.quasi_dists[0].get('1') == None:
                    inter = 0.
                else:
                    inter = res_.quasi_dists[0].get('1')
                thets = pd.DataFrame(np.array([thetas]), index=np.array([0]))
                thets.to_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/th_{shots}_{a}_{dt}_{tf}.csv')
                doub_occ = pd.DataFrame(np.array([(1 - (1 - 2 * inter))/4]), index=np.array([0]))
                doub_occ.to_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/do_{shots}_{a}_{dt}_{tf}.csv')
                ts = thets.index
                
            while ts[-1] < tf:
                
                START = timer()
                
                # Assigning the correct parameters to each circuit explicitly.
                param_vec = [thetas[0:len(circ.parameters)] for circ in all_circuits]
                res = sampler(circuits=[i for i in range(len(all_circuits))], 
                               parameter_values=[param_vec[i] for i in range(len(param_vec))],
                              shots=shots)

                # Evaluate the double occupancy of the PREVIOUS STEP if thets.index[-1] is larger
                # than the doub_occ.index[-1] (i.e., if the program has already been running, but
                # not reached tf yet). If the program does reach tf, then it will compute the double
                # occupancy at the end (which requires submitting a new circuit to the queue). This
                # is necessary 
                if thets.index[-1] > doub_occ.index[-1]:
                    if res.quasi_dists[-1].get('1') == None:
                        inter = 0.
                    else:
                        inter = res.quasi_dists[-1].get('1')
                    doub_occ.loc[ts[-1]] = np.array((1 - (1 - 2 * inter))/4)
                    doub_occ.to_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/do_{shots}_{a}_{dt}_{tf}.csv')

                # Making the M matrix

                M = np.zeros((len(tee), len(tee)))
                for i, m, n in m_loc:
                    if res.quasi_dists[i].get('1') == None:
                        inter = 0.
                    else:
                        inter = res.quasi_dists[i].get('1')
                    M[m, n] = 2 * (1 - 2 * inter)
                    if n > m:
                        M[n, m] = M[m, n]

                # Making Delta

                if res.quasi_dists[-2].get('1') == None:
                    inter = 0.
                else:
                    inter = res.quasi_dists[-2].get('1')
                Delta = (1 - 2 * inter)/4

                # With Delta, can make V

                hiVal = Uc/4
                hxVal = -Uc*Delta
                hzVal = -Uf/4
                V = np.zeros(len(tee))
                for m in range(len(tee)):
                    vee = 0
                    for i, ham_type, n in v_loc:
                        if n == m:
                            if ham_type == 'X':
                                if res.quasi_dists[i].get('1') == None:
                                    inter = 0.
                                else:
                                    inter = res.quasi_dists[i].get('1')
                                vee += hxVal * (1 - 2 * inter)
                            elif ham_type == 'Z':
                                if res.quasi_dists[i].get('1') == None:
                                    inter = 0.
                                else:
                                    inter = res.quasi_dists[i].get('1')
                                vee += hzVal * (1 - 2 * inter)
                            elif ham_type == 'I':
                                if res.quasi_dists[i].get('1') == None:
                                    inter = 0.
                                else:
                                    inter = res.quasi_dists[i].get('1')
                                vee += hiVal * (1 - 2 * inter)
                    V[m] = 2. * vee
                
                # Qiskit time evolution has e^{-iHt/2}, so need a multiplication by 2 here.
                dtheta = (lsq_linear(M,V,(-b,b)).x) * 2 * dt
                thetas = list(np.array(thetas[:]) + dtheta)
                
                thets.loc[ts[-1] + dt] = np.array(thetas[:])
                thets.to_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/th_{shots}_{a}_{dt}_{tf}.csv')
                ts = thets.index
                if ts[-1] > tf:
                    param_do_end = thetas[0:len(all_circuits[-1].parameters)]
                    _res_ = sampler(circuits=[len(all_circuits)-1], 
                               parameter_values=[param_do_end],
                              shots=shots)
                    if _res_.quasi_dists[0].get('1') == None:
                        inter = 0.
                    else:
                        inter = _res_.quasi_dists[0].get('1')
                    doub_occ.loc[ts[-1]] = np.array((1 - (1 - 2 * inter))/4)
                    doub_occ.to_csv(f'./resQT_{shots}_{a}_{dt}_{tf}/do_{shots}_{a}_{dt}_{tf}.csv')
                
                
                print(timer() - START)
                
        print('Done.')

    except KeyboardInterrupt:
        print('Exiting.')
    except:
        # If it throws an error because the Runtime API kicked me out, run it again.
        os.system(f'python3 hm_runtime.py -a {a} -b {b} -t {tf} -d {dt} -s {shots}')



