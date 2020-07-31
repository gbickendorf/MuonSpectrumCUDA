# MuonSpectrumCUDA

This project is a large part of my master thesis. 

If certain dark matter models were present additionally to the Standard
Model that couple exclusively to leptons, these would avoid most
experimental constraints. If these particles were to decay either slowly
or invisibly, the only effect visible to decay experiments is a slight
deformation of the decay-energy-spectrum.

The electron spectrum in muon decays is well known and is measured with
high precision. Thus there exists an upper bound on the deformation which 
can be used to derive constraints on the parameter space of possible dark
matter models.

Simulating the effect of additional mediators by run of the mill solutions
such as madgraph proves unviable, since the number of events necessary to 
simulate experimental precision is to large. 

Here a custom numerical integration scheme is needed. The fine experimental
grid of the spectrum mandates a large number of nodes which prove
very paralelisable. The CUDA-Framework fits this task perfectly, reducing the
time to get sufficient precision to mere minutes.
