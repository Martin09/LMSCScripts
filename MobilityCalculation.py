"""
Calculates the mobility of a nanowire from the geometry, measured resistance and estimated carrier concentration
Written by Martin Friedl
01/11/2016
"""

import pint
ureg = pint.UnitRegistry()

# Define the inputs
R = (27.34 * ureg.kohm).plus_minus(0.1)  # Resistance of nanowire
L = (0.828 * ureg.um).plus_minus(0.1)  # Length of nanowire
W = (87 * ureg.nm).plus_minus(10)  # Width of nanowire
H = (50 * ureg.nm).plus_minus(10)  # Height of nanowire
n = (2.2E20 * ureg.cm**-3).plus_minus(1E20)  # Doping concentration
e = ureg.e

# Calculate the values
rho = R * W * H / L  # Calculate resistivity
print('The resistivity is {}'.format(rho.to(ureg.ohm*ureg.cm)))  # Change units to ohm*cm

mu = 1/(e*rho*n)  # Calculate mobility
print('The mobility is {}'.format(mu.to(ureg.cm**2/ureg.V/ureg.s)))  # Change units to ohm*cm
