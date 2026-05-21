from metpy.units import units
from metpy import constants

H = 7 * units.km                    # [km] scale height
P0 = 1000 * units.hPa               # [hPa] ground pressure
gEarth = constants.earth_gravity    # [m/s^2]  standart gravity
R = constants.dry_air_gas_constant  # universal gas constant for dry air [J kg^-1 K^-1]
Cp = constants.dry_air_spec_heat_press # Specific heat at constant pressure for dry air [J kg^-1 K^-1]
Ts = 240 * units.K                 # [K] Constant reference temperature in middle atmosphere it is common to let it 240K
rEarth = constants.earth_avg_radius     # [m] mean radius of the earth
angVeloEarth = constants.earth_avg_angular_vel  # [rad/s] angular velocity of the Earth (2*np.pi)/(23*60*60-236) 

