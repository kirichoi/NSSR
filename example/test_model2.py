# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 03:00:22 2017

@author: KIRI
"""

import tellurium as te
import roadrunner

#r1 = te.loada("""
#S1 + S2 -> S3 + S4; k1*S1*S2;
#
#S1 = 1.0; S2 = 2.0; S3 = 0.1; S4 = .5;
#k1 = 0.1
#""")
#
#rr1 = r1.simulate(0, 100, 100)
#r1.plot()
#
#r2 = te.loada("""
#S1 -> S3; k1*S1;
#S1 -> S4; k2*S1;
#S2 -> S3; k3*S2;
#S2 -> S4; k4*S2;
#
#S1 = 1.0; S2 = 2.0; S3 = 0.1; S4 = .5;
#k1 = 0.05; k2 = 0.05; k3 = 0.05; k4 = 0.05;
#""")
#
#rr2 = r2.simulate(0, 100, 100)
#r2.plot()

r3 = te.loada("""
S1 -> S2; k1*S1/(S4 + 1);
S2 -> S1; k2*S2;
S3 -> S4; k3*S2*S3;
S4 -> S3; k4*S4;

S1 = .83; S2 = 2.0; S3 = 0.1; S4 = .5
k1 = 0.05; k2 = 0.05; k3 = 0.05; k4 = 0.05;
""")

rr3 = r3.simulate(0, 100, 100)
print(rr3[-1])