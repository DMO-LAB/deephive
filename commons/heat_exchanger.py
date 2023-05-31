import numpy as np

class HeatExchanger:
    """ 
    A Mathematical Model for Heat Exchanger. The objective function is the total annual cost of operating a shell and tube
    heat exchanger. The cost is calculated some reference equations presented in the report.
    """
    def __init__(self, params):
        self.params = params
 
    def cal_htc(self, Re: np.ndarray) -> np.ndarray:
        """
        Calculate the heat transfer coefficient for a given Reynolds number
        :param Re: Reynolds number
        :return: htc -> heat transfer coefficient
        """
        params = self.params
        htc = np.zeros_like(Re)
        mask1 = Re < 2300
        mask2 = np.logical_and(Re >= 2300, Re < 10000)
        mask3 = Re >= 10000
        if np.sum(mask1) > 0:
            if Re[mask1].shape != sum(mask1):
                raise ValueError('Invalid input shape for htc1 function')
            htc1 = lambda Ree: (params['tube']['kt'] / params['di'][mask1]) * (3.657 + 0.0677 * Ree * params['Prt'] \
                * ((params['di'][mask1] / params['L'])**(1/3))) / (1 + 0.1 * params['Prt'] * (Ree * (params['di'] / params['L'])**(1/3)))
            htc[mask1] = htc1(Re[mask1])
            
        if np.sum(mask2) > 0:
            if Re[mask2].shape != sum(mask2):
                raise ValueError('Invalid input shape for htc2 function')
            htc2 = lambda Ree: (params['tube']['kt'] / params['di'][mask2]) * (((1 + params['di'][mask2] / params['L'])**0.67) \
                * ((params['ft'][mask2] / 8) * (Ree - 1000) * params['Prt']) / \
                    (1 + 12.7 * (params['ft'][mask2] / 8)**0.5 * (params['Prt']**(2/3) - 1)))
            htc[mask2] = htc2(Re[mask2])
            
        if np.sum(mask3) > 0:
            if Re[mask3].shape != sum(mask3):
                raise ValueError('Invalid input shape for htc3 function')
            htc3 = lambda Ree: 0.027 * (params['tube']['kt'] / params['di'][mask3]) * Ree**0.8 * params['Prt']**(1/3) \
                * (params['tube']['mut'] / params['tube']['muwt'])**0.14
            htc[mask3] = htc3(Re[mask3])

        return htc

    def objective_function(self, xx: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Calculate the objective function for a given set of parameters
        :param xx: parameters to be optimized
            xx[:, 0] -> tube outside diameter [m]
            xx[:, 1] -> shell inside diameter [m]
            xx[:, 2] -> baffles spacing [m]
        :param debug: print debug information
        :return: Ctotal -> total cost of operating a shell and tube heat exchanger for a given number of years
        """
        #print(f"xx: {xx}")
        st = 1.25 * xx[:, 0]
        di = 0.8 * xx[:, 0]
        self.params['di'] = di
        self.params['Nt'] = self.params['constants']['C'] * (xx[:, 1]/xx[:, 0])**self.params['constants']['n1']
        vt = self.params['tube']['mt'] / (self.params['tube']['rhot']
                                          * np.pi * di**2 / 4) * (self.params['n'] / self.params['Nt'])
        Ret = self.params['tube']['rhot'] * vt * di / \
                self.params['tube']['mut']  # Reynolds number
        Prt = self.params['tube']['cpt'] * self.params['tube']['mut'] / \
                self.params['tube']['kt']  # Prandtl number
        ft =  (1.82 * np.log10(Ret) - 1.64)**(-2)  # Darcy friction factor
        self.params['ft'] = ft
        de = 4 * (0.43 * st**2 - (0.5*np.pi *
                  xx[:,0]**2 / 4)) / (0.5 * np.pi * xx[:, 0])
        As = xx[:, 1] * xx[:,2] * (1 - xx[:, 0] / st)  # shell side heat transfer area
        vs = self.params['shell']['ms'] / \
                (self.params['shell']['rhos'] * As)  # shell side velocity
        Res = self.params['shell']['ms'] * de / \
                (self.params['shell']['mews'] * As)  # Reynolds number
        Prs = self.params['shell']['cps'] * self.params['shell']['mews'] / \
                self.params['shell']['ks']  # Prandtl number

        # shell side heat transfer coefficient
        hs = 0.36 * (self.params['shell']['ks'] / de) * Res**0.55 * Prs**(1/3) * (
                self.params['shell']['mews'] / self.params['shell']['mewws'])**0.14
        R = (self.params['shell']['Thi'] - self.params['shell']['Tho']
                ) / (self.params['tube']['Tco'] - self.params['tube']['Tci'])
        P = (self.params['tube']['Tco'] - self.params['tube']['Tci']) / \
                (self.params['shell']['Thi'] - self.params['tube']['Tci'])
        
        f1 = np.sqrt((R**2) + 1) / (R - 1) 
        f2 = np.log((1 - P)/ (1 - P*R))
        f3 = ((2/P) - 1 - R + np.sqrt(R**2 + 1)) / ((2/P) - 1 - R - np.sqrt(R**2 + 1))
        F = f1 * (f2 / np.log(f3))
  
        LMTD = (self.params['shell']['Thi'] - self.params['tube']['Tco']) - (self.params['shell']['Tho'] - self.params['tube']['Tci']) / \
            np.log((self.params['shell']['Thi'] - self.params['tube']['Tco']) /
                   (self.params['shell']['Tho'] - self.params['tube']['Tci']))
        # Q = self.params['tube']['mt'] * self.params['tube']['cpt'] * \
        #     (self.params['tube']['Tco'] - self.params['tube']['Tci'])
        Q = self.params['shell']['ms'] * self.params['shell']['cps'] * \
            (self.params['shell']['Thi'] - self.params['shell']['Tho'])
        self.params['di'] = di
        self.params['Prt'] = Prt

        ht = self.cal_htc(Ret)
        U = 1 / ((1 / hs) + (self.params['shell']['Rfs']) + ((self.params['tube']['Rft']) + (1 / ht)) * (xx[:, 0]/di))
        A = Q / (U * LMTD * F)
        L = A / (np.pi * di * self.params['Nt'])  # tube length calculated
        #L = self.params['L']

        # pump power
        p = self.params['constants']['p']
        pt = (0.5 * self.params['tube']['rhot'] * vt**2) * ((L * ft/(di)) + p) * self.params['n']
        fs = 1.44 * Res**(-0.15)  # shell side friction factor
        ps = fs * ((self.params['shell']['rhos'] * vs **2) / 2) * (L / xx[:,2]) * (xx[:, 1] / de)

        # Objective function : Total annual cost of operating a shell and tube heat exchanger
        Ci = self.params['constants']['a1'] + \
            self.params['constants']['a2'] * A**self.params['constants']['a3'] # Ci : Capital investment cost
        
        P1 = ((self.params['tube']['mt'] * pt/self.params['tube']['rhot']) + (
            self.params['shell']['ms'] * ps/self.params['shell']['rhos'])) / self.params['eff']
        ce = self.params['ce'] # ce : cost of electricity
        H = self.params['H'] # H : hours of operation per year
        Co = ce * H * P1 # Co : Annual Operating cost
        ny = self.params['ny'] # ny : number of years of operation
        j = np.linspace(1, ny, 10)
        i = self.params['constants']['i']
        Cod =  np.array([np.sum(Coi * (1 + i)**(j)) for Coi in Co])  # Cod : Total discounted operating cost
        Ctotal = Ci + Cod # Ctotal : Total annual cost of operating a shell and tube heat exchanger
        self.parameters = {
            'do':xx[:,0], "Ds":xx[:,1], "B":xx[:,2],
            'st': st, 'di': di, 'vt': vt, 'Ret': Ret, 'Prt': Prt, 'Nt': self.params['Nt'],
                'ft': ft, 'de': de, 'As': As, 'vs': vs, 'Res': Res, 
                    'Prs': Prs, 'hs': hs, 'R': R, 'P': P, 'F': F, 'LMTD': LMTD, 
                        'Q': Q, 'ht': ht, 'U': U, 'A': A, 'L': L, 'pt': pt, 'fs': fs, 
                            'ps': ps, 'Ci': Ci, 'P1': P1, 'Co': Co, 'Cod': Cod, 'Ctotal': Ctotal, 'Ln': L}
        if debug:
            print(self.parameters)

        return -Ctotal

    def get_parameters(self, quantity):
        if quantity in self.parameters.keys():
            return self.parameters[quantity]
        else:
            raise ValueError('The quantity is not available.')

    def __call__(self, xx, debug=False):
        return self.objective_function(xx, debug)

    def __str__(self):
        return f"A shell and tube Heat Exchanger with Model"

    def constraints(self, xx):
        """ 
        xx[:, 0] should be between 0.015m to 0.051m
        xx[:, 1] should be between 0.1m to 1.5m
        xx[:, 2] should be between 0.05m to 0.5m
        """
        constraints = []
        if xx[:, 0] < 0.015 or xx[:, 0] > 0.051:
            constraints.append(False)
        else:
            constraints.append(True)
        if xx[:, 1] < 0.1 or xx[:, 1] > 1.5:
            constraints.append(False)
        else:
            constraints.append(True)
        if xx[:,2] < 0.05 or xx[:,2] > 0.5:
            constraints.append(False)
        else:
            constraints.append(True)
        return constraints




# Operating parameters, fluid properties, cost parameters and other constants
params = {
    'tube': {
        'mt': 68.90,  # mass flow rate of tube side fluid (kg/s)
        'Tci': 25,  # cold fluid inlet temperature (C)
        'Tco': 40,  # cold fluid outlet temperature (C)
        'rhot': 995.0,  # density of tube side fluid (kg/m^3)
        'mut': 0.0008,  # viscosity of tube side fluid (kg/m.s)
        'muwt': 0.00052,# viscosity of tube side fluid at wall temperature (kg/m.s)
        'cpt': 4.2 * 1000,  # specific heat capacity of tube side fluid (kJ/kg.K)
        'kt': 0.59,  # thermal conductivity of tube side fluid (W/m.K)
        'Rft': 0.0002,  # fouling resistance of tube side fluid (m^2.K/W)
    },
    'shell': {
        'ms': 27.8,  # mass flow rate of shell side fluid (kg/s)
        'Thi': 95,  # hot fluid inlet temperature (C)
        'Tho': 40,  # hot fluid outlet temperature (C)
        'rhos': 750,  # density of shell side fluid (kg/m^3)
        'cps': 2.84 * 1000,  # specific heat capacity of shell side fluid (kJ/kg.K)
        'mews': 0.00034,  # viscosity of shell side fluid (kg/m.s)
        'mewws': 0.00038, # viscosity of shell side fluid at wall temperature (kg/m.s)
        'ks': 0.19,  # thermal conductivity of shell side fluid (W/m.K)
        'Rfs': 0.00033,  # fouling resistance of shell side fluid (m^2.K/W)
    },
    'constants': {
        'a1':8000,  # numerical constant
        'a2':259.2,  # numerical constant
        'a3':0.91,  # numerical constant
        'p': 4,  # constant
        'i': 0.1, # constant
        'C': 0.319, # constant
        'n1': 2.142, # constant
    },
    'n': 2,  # number of passes
    'eff': 0.8,  # pump efficiency
    'L': 3.115,  # tube length (m)
    'ny': 10, # number of years
    'H': 7000, # number of hours
    'ce': 0.00012 # cost of electricity per kWh

}


HeatExchangerModel = HeatExchanger(params)