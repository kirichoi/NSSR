# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:45:10 2017

@author: Kiri Choi
"""

import roadrunner
import tellurium as te
import psutil
import antimony


ant_str = '''
model synth()

$S0 -> S1; k1*S0;
S1 -> S2; k2*S1/(1 + S5);
S2 -> S3; k3*S2;
S3 -> S4; k4*S3/(1 + S5);
S4 -> S5; k5*S4;
S5 -> $S6; k6*S5;

$S0 = 1; S1 = 0.35; S2 = 0.2; S3 = 0.15; S4 = 0.2; S5 = 0.15; $S6 = 1; 
k1 = 0.6; k2 = 0.7; k3 = 0.2; k4 = 0.2; k5 = 0.25; k6 = 0.4; 

end
'''


#ant_str = '''
#model synth()
#
#J0: $X0 + S1 => S2; k1*X0*S1;
#J1: $X0 + S2 => S3; k2*X0*S2;
#J2: S2 => S1; k3*S2;
#J3: S3 => S2; k4*S3;
#J4: S3 + S4 => S3 + S5; k5*S3*S4;
#J5: S3 + S5 => S3 + S6; k6*S3*S5;
#J6: S5 => S4; k7*S5;
#J7: S6 => S5; k8*S6;
#J8: S6 => $X1; k9*S6;
#
#X0 = 1; X1 = 0.1; S1 = 0.35; S2 = 0.2; S3 = 0.15; S4 = 0.2; S5 = 0.15; S6 = 0.1;
#k1 = 0.6; k2 = 0.7; k3 = 0.2; k4 = 0.2; k5 = 0.25; k6 = 0.4; 
#k7 = 0.35; k8 = 0.15; k9 = 0.55;
#
#end
#'''

r = te.loada(ant_str)
result = r.simulate(0, 200, 200)
r.plot()
print(r.steadyStateNamedArray())
