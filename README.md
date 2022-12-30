# Hubbard_model_IBM_device

This repository is for scripts that run simulations for the Hubbard model at half-filling. At half-filling we can write the Hubbard model in terms of a spin Hamiltonian. The simulations are single-spin Hamiltonians, with the idea to run them on a real IBMQ device. For parameterized time evolution we use the McLachlan variational principle.

There are three files: 

noNoiseHM.py    :: This file runs the time evolution locally, without a noise model (uses QASM).

noiseModelHM.py :: This file runs the time evolution locally, with a noise model derived from a device.

hm_runtime.py   :: This file uses the IBMQ runtime interface to run the time evolution on a real IBM device (ibmq_quito).
