from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SEIARV():
    """
    Suspected-Exposed-Symptomatic Infected-Asymptomatic Infected
    -Recovered-Vaccinated Model
    """

    def __init__(self, ctrl_C=None, ctrl_D=None, waning_rate=None, water_input_C=None, water_input_D=None,
                 travel=None, travel_V=None, vac_plan_C=None, vac_plan_D=None,
                 initS_C=100, initE_C=1, initA_C=1, initI_C=1, initR1_C=0, initR2_C=0,
                 initV_C=0, initE_C_V=0, initA_C_V=0, initI_C_V=0, initR1_C_V=0, initR2_C_V=0,
                 initW_C=0, initM_C=100,
                 initS_D=1000, initE_D=1, initA_D=1, initI_D=1, initR1_D=0, initR2_D=0,
                 initV_D=0, initE_D_V=0, initA_D_V=0, initI_D_V=0, initR1_D_V=0, initR2_D_V=0,
                 initW_D=0, initM_D=100,
                 t0=0, dt=0.1, tend=200,
                 N_C=100, N_D=200,
                 beta_C=0.2, beta_D=0.22,
                 k_E=1 / 3, k_A=1 / 3, sigma=1 / 5.2, rho=0.18,
                 gamma_I=1 / 12, gamma_A=1 / 10, gamma_R=1 / 13,
                 p_0=100, p_E=0.1, p_A=0.1, p_R=0.1,
                 lambda_C=1, lambda_D=1,
                 nu=0.2, epsilon=0.5, rho_V=0.8, sigma_V=1 / 5, gamma_I_V=1 / 10, gamma_A_V=1 / 8, gamma_R_V=1 / 10,
                 p_1=50,
                 nature_birth_C=0, nature_birth_D=0, nature_death=0, vac_rate=0):
        self.ctrl_C = ctrl_C if ctrl_C is not None else lambda t: 1
        self.ctrl_D = ctrl_D if ctrl_D is not None else lambda t: 1
        self.waning_rate = waning_rate if waning_rate is not None else lambda t: 1
        self.water_input_C = water_input_C if water_input_C is not None else lambda t: 100
        self.water_input_D = water_input_D if water_input_D is not None else lambda t: 100
        self.travel = travel if travel is not None else lambda t: 0
        self.travel_V = travel_V if travel_V is not None else lambda t: 0
        self.vac_plan_C = vac_plan_C if vac_plan_C is not None else lambda t: 0
        self.vac_plan_D = vac_plan_D if vac_plan_D is not None else lambda t: 0
        # time-dependent function; Initial value; time; parameters; others
        self.initvalue = {"initS_C": initS_C, "initE_C": initE_C, "initA_C": initA_C, "initI_C": initI_C,
                          "initR1_C": initR1_C, "initR2_C": initR2_C,
                          "initV_C": initV_C, "initE_C_V": initE_C_V, "initA_C_V": initA_C_V, "initI_C_V": initI_C_V,
                          "initR1_C_V": initR1_C_V, "initR2_C_V": initR2_C_V, "initW_C": initW_C, "initM_C": initM_C,
                          "initS_D": initS_D, "initE_D": initE_D, "initA_D": initA_D, "initI_D": initI_D,
                          "initR1_D": initR1_D, "initR2_D": initR2_D,
                          "initV_D": initV_D, "initE_D_V": initE_D_V, "initA_D_V": initA_D_V, "initI_D_V": initI_D_V,
                          "initR1_D_V": initR1_D_V, "initR2_D_V": initR2_D_V, "initW_D": initW_D, "initM_D": initM_D}
        self.timepara = {"t0": t0, "dt": dt, "tend": tend}
        self.para = {"N_C": N_C, "N_D": N_D,
                     "beta_C": beta_C, "beta_D": beta_D,
                     "k_E": k_E, "k_A": k_A, "sigma": sigma, "rho": rho,
                     "gamma_I": gamma_I, "gamma_A": gamma_A, "gamma_R": gamma_R,
                     "p_0": p_0, "p_E": p_E, "p_A": p_A, "p_R": p_R,
                     "lambda_C": lambda_C, "lambda_D": lambda_D,
                     "nu": nu, "epsilon": epsilon, "rho_V": rho_V, "sigma_V": sigma_V,
                     "gamma_I_V": gamma_I_V, "gamma_A_V": gamma_A_V, "gamma_R_V": gamma_R_V, "p_1": p_1,
                     "nature_birth_C": nature_birth_C, "nature_birth_D": nature_birth_D, "nature_death": nature_death,
                     "vac_rate": vac_rate}

    # Compartment models
    @staticmethod
    def SEIARV_model(y, t, N_C=100, N_D=200,
                     beta_C=0.2, beta_D=0.22,
                     k_E=1 / 3, k_A=1 / 3, sigma=1 / 5.2, rho=0.18,
                     gamma_I=1 / 12, gamma_A=1 / 10, gamma_R=1 / 13,
                     p_0=100, p_E=0.1, p_A=0.1, p_R=0.1,
                     lambda_C=1, lambda_D=1,
                     nu=0.2, epsilon=0.5, rho_V=0.8, sigma_V=1 / 5, gamma_I_V=1 / 10, gamma_A_V=1 / 8, gamma_R_V=1 / 10,
                     p_1=50, nature_birth_C=0, nature_birth_D=0, nature_death=0, vac_rate=0):
        S_C, E_C, A_C, I_C, R1_C, R2_C, V_C, E_C_V, A_C_V, I_C_V, R1_C_V, R2_C_V, W_C, M_C, \
        S_D, E_D, A_D, I_D, R1_D, R2_D, V_D, E_D_V, A_D_V, I_D_V, R1_D_V, R2_D_V, W_D, M_D = y

        # time-dependent function
        SEIARV_1 = SEIARV()
        ctrl_C = SEIARV_1.ctrl_C  # input control function
        ctrl_D = SEIARV_1.ctrl_D
        waning_rate = SEIARV_1.waning_rate
        water_input_C = SEIARV_1.water_input_C
        water_input_D = SEIARV_1.water_input_D
        travel = SEIARV_1.travel
        travel_V = SEIARV_1.travel_V
        vac_plan_C = SEIARV_1.vac_plan_C
        vac_plan_D = SEIARV_1.vac_plan_D
        # Calculation
        N1_C = S_C + E_C + A_C + I_C + R1_C + R2_C
        N2_C = V_C + E_C_V + A_C_V + I_C_V + R1_C_V + R2_C_V
        N_stay_C = (1 - travel(t)) * N1_C + travel(t) * I_C + (1 - travel_V(t)) * N2_C + travel_V(t) * I_C_V
        N_stay_D = N_D + travel(t) * N1_C - travel(t) * I_C + travel_V(t) * N2_C - travel_V(t) * I_C_V

        ## non-vac individuals with infectiveness in University
        Infective_C = k_E * (1 - travel(t)) * E_C + k_A * (
                1 - travel(t)) * A_C + I_C
        ## vac individuals with infectiveness in University
        Infective_C_V = k_E * (1 - travel_V(t)) * E_C_V + k_A * (
                1 - travel_V(t)) * A_C_V + I_C_V
        ## non-vac individuals with infectiveness in Downtown
        Infective_D = k_E * (travel(t) * E_C + E_D) + k_A * (travel(t) * A_C + A_D) + I_D
        ## vac individuals with infectiveness in Dwontown
        Infective_D_V = k_E * (travel_V(t) * E_C_V + E_D_V) + k_A * (travel_V(t) * A_C_V + A_D_V) + I_D_V
        ## non-vac infection of students in university
        Infec_C = beta_C * ctrl_C(t) * (1 - travel(t)) * S_C * (Infective_C + nu * Infective_C_V) / N_stay_C
        ## non-vac infection of students in downtown
        Infec_D = beta_D * ctrl_D(t) * travel(t) * S_C * (Infective_D + nu * Infective_D_V) / N_stay_D
        ## vac infection of students in university
        Infec_C_V = epsilon * beta_C * ctrl_C(t) * (1 - travel_V(t)) * V_C * (
                Infective_C + nu * Infective_C_V) / N_stay_C
        ## vac infection of students in downtown
        Infec_D_V = epsilon * beta_D * ctrl_D(t) * travel_V(t) * V_C * (Infective_D + nu * Infective_D_V) / N_stay_D
        ## non-vac infection of downtown individuals
        down_Infec = beta_D * ctrl_D(t) * S_D * (Infective_D + nu * Infective_D_V) / N_stay_D
        ## vac infection of downtown individuals
        down_Infec_V = epsilon * beta_D * ctrl_D(t) * V_D * (Infective_D + nu * Infective_D_V) / N_stay_D
        ## Viral shedding in University
        shed_C = p_0 * (I_C + (1 - travel(t)) * (p_A * A_C + p_E * E_C + p_R * R1_C)) + p_1 * (
                I_C_V + (1 - travel_V(t)) * (p_A * A_C_V + p_E * E_C_V + p_R * R1_C_V))
        shed_D = p_0 * travel(t) * (p_A * A_C + p_E * E_C + p_R * R1_C) + p_1 * travel_V(t) * (
                p_A * A_C_V + p_E * E_C_V + p_R * R1_C_V) + p_0 * (
                         I_D + p_A * A_D + p_E * E_D + p_R * R1_D) + p_1 * (
                         I_D_V + p_A * A_D_V + p_E * E_D_V + p_R * R1_D_V)

        ## Choice of vaccination term phi(t)S or phi(t); vac_rate=0, phi(t); vac_rate=1, phi(t) S
        vac_C = vac_plan_C(t) * (vac_rate * S_C + 1 - vac_rate)
        vac_D = vac_plan_D(t) * (vac_rate * S_D + 1 - vac_rate)

        # dydt
        dS_C = nature_birth_C - Infec_C - Infec_D - vac_C - nature_death * S_C
        dE_C = Infec_C + Infec_D - sigma * E_C - nature_death * E_C
        dA_C = rho * sigma * E_C - gamma_A * A_C - nature_death * A_C
        dI_C = (1 - rho) * sigma * E_C - gamma_I * I_C - nature_death * I_C
        dR1_C = gamma_A * A_C + gamma_I * I_C - gamma_R * R1_C - nature_death * R1_C
        dR2_C = gamma_R * R1_C - nature_death * R2_C
        dV_C = vac_C - Infec_C_V - Infec_D_V - nature_death * V_C
        dE_C_V = Infec_C_V + Infec_D_V - sigma_V * E_C_V - nature_death * E_C_V
        dA_C_V = rho_V * sigma_V * E_C_V - gamma_A_V * A_C_V - nature_death * A_C_V
        dI_C_V = (1 - rho_V) * sigma_V * E_C_V - gamma_I_V * I_C_V - nature_death * I_C_V
        dR1_C_V = gamma_A_V * A_C_V + gamma_I_V * I_C_V - gamma_R_V * R1_C_V - nature_death * R1_C_V
        dR2_C_V = gamma_R_V * R1_C_V - nature_death * R2_C_V
        dW_C = shed_C / M_C - waning_rate(t) * W_C
        dM_C = water_input_C(t) - lambda_C * M_C
        dS_D = nature_birth_D - down_Infec - vac_D - nature_death * S_D
        dE_D = down_Infec - sigma * E_D - nature_death * E_D
        dA_D = rho * sigma * E_D - gamma_A * A_D - nature_death * A_D
        dI_D = (1 - rho) * sigma * E_D - gamma_I * I_D - nature_death * I_D
        dR1_D = gamma_A * A_D + gamma_I * I_D - gamma_R * R1_D - nature_death * R1_D
        dR2_D = gamma_R * R1_D - nature_death * R2_D
        dV_D = vac_D - down_Infec_V - nature_death * V_D
        dE_D_V = down_Infec_V - sigma_V * E_D_V - nature_death * E_D_V
        dA_D_V = rho_V * sigma_V * E_D_V - gamma_A_V * A_D_V - nature_death * A_D_V
        dI_D_V = (1 - rho_V) * sigma_V * E_D_V - gamma_I_V * I_D_V - nature_death * I_D_V
        dR1_D_V = gamma_A_V * A_D_V + gamma_I_V * I_D_V - gamma_R_V * R1_D_V - nature_death * R1_D_V
        dR2_D_V = gamma_R_V * R1_D_V - nature_death * R2_D_V
        dW_D = shed_D / M_D - waning_rate(t) * W_D
        dM_D = water_input_D(t) - lambda_D * M_D
        return np.array(
            [dS_C, dE_C, dA_C, dI_C, dR1_C, dR2_C, dV_C, dE_C_V, dA_C_V, dI_C_V, dR1_C_V, dR2_C_V, dW_C, dM_C,
             dS_D, dE_D, dA_D, dI_D, dR1_D, dR2_D, dV_D, dE_D_V, dA_D_V, dI_D_V, dR1_D_V, dR2_D_V, dW_D, dM_D])

    # Solving the model by odeint
    def ode_sol(self):
        init_value = [self.initvalue[i] for i in self.initvalue.keys()]
        # print("Initial Value:", init_value)
        tspan = np.arange(self.timepara["t0"], self.timepara["tend"], self.timepara["dt"])  # time span
        # print("Tspan:", tspan)
        para = tuple([self.para[i] for i in self.para.keys()])  # args
        # print("Parameters:", para)
        sol = odeint(self.SEIARV_model, init_value, tspan, para, )
        return {"tspan": tspan, "solution": sol}


class SEIARV_one():
    """
    One region Suspected-Exposed-Symptomatic Infected-Asymptomatic Infected
    -Recovered-Vaccinated Model
    """

    def __init__(self, ctrl=None, waning_rate=None, water_input=None,
                 vac_plan=None, diagnosed=None, diagnosed_V=None,
                 initS=100, initE=1, initA=1, initI=1, initD=0, initR1=0, initR2=0,
                 initV=0, initE_V=0, initA_V=0, initI_V=0, initD_V=0, initR1_V=0, initR2_V=0,
                 initW=0, initM=100,
                 t0=0, dt=0.1, tend=200,
                 N=100,
                 beta=0.2,
                 k_E=1 / 3, k_A=1 / 3, sigma=1 / 5.2, rho=0.18,
                 gamma_I=1 / 12, gamma_A=1 / 10, gamma_R=1 / 13, gamma_D=1 / 10,
                 p_0=100, p_E=0.1, p_A=0.1, p_R=0.1,
                 lambda_0=1,
                 nu=0.2, epsilon=0.5, rho_V=0.8, sigma_V=1 / 5, gamma_I_V=1 / 10, gamma_A_V=1 / 8, gamma_R_V=1 / 10,
                 gamma_D_V=1 / 10,
                 p_1=50,
                 nature_birth=0, nature_death=0, vac_rate=0, diag_rate=1):
        self.ctrl = ctrl if ctrl is not None else lambda t: 1
        self.waning_rate = waning_rate if waning_rate is not None else lambda t: 1
        self.water_input = water_input if water_input is not None else lambda t: 100
        self.vac_plan = vac_plan if vac_plan is not None else lambda t: 0
        self.diagnosed = diagnosed if diagnosed is not None else lambda t: 0
        self.diagnosed_V = diagnosed_V if diagnosed_V is not None else lambda t: 0
        # time-dependent function; Initial value; time; parameters; others
        self.initvalue = {"initS": initS, "initE": initE, "initA": initA, "initI": initI, "initD": initD,
                          "initR1": initR1, "initR2": initR2,
                          "initV": initV, "initE_V": initE_V, "initA_V": initA_V, "initI_V": initI_V,
                          "initD_V": initD_V,
                          "initR1_V": initR1_V, "initR2_V": initR2_V, "initW": initW, "initM": initM}
        self.timepara = {"t0": t0, "dt": dt, "tend": tend}
        self.para = {"N": N,
                     "beta": beta,
                     "k_E": k_E, "k_A": k_A, "sigma": sigma, "rho": rho,
                     "gamma_I": gamma_I, "gamma_A": gamma_A, "gamma_R": gamma_R, "gamma_D": gamma_D,
                     "p_0": p_0, "p_E": p_E, "p_A": p_A, "p_R": p_R,
                     "lambda_0": lambda_0,
                     "nu": nu, "epsilon": epsilon, "rho_V": rho_V, "sigma_V": sigma_V,
                     "gamma_I_V": gamma_I_V, "gamma_A_V": gamma_A_V, "gamma_R_V": gamma_R_V, "gamma_D_V": gamma_D_V,
                     "p_1": p_1,
                     "nature_birth": nature_birth, "nature_death": nature_death,
                     "vac_rate": vac_rate, "diag_rate": diag_rate}

    # Compartment models
    @staticmethod
    def SEIARV_one_model(y, t, N=100,
                         beta=0.2,
                         k_E=1 / 3, k_A=1 / 3, sigma=1 / 5.2, rho=0.18,
                         gamma_I=1 / 12, gamma_A=1 / 10, gamma_R=1 / 13, gamma_D=1 / 10,
                         p_0=100, p_E=0.1, p_A=0.1, p_R=0.1,
                         lambda_0=1,
                         nu=0.2, epsilon=0.5, rho_V=0.8, sigma_V=1 / 5, gamma_I_V=1 / 10, gamma_A_V=1 / 8,
                         gamma_D_V=1 / 10,
                         gamma_R_V=1 / 10,
                         p_1=50, nature_birth=0, nature_death=0, vac_rate=0, diag_rate=1):
        S, E, A, I, D, R1, R2, V, E_V, A_V, I_V, D_V, R1_V, R2_V, W, M, = y

        # time-dependent function
        SEIARV_1 = SEIARV_one()
        ctrl = SEIARV_1.ctrl  # input control function
        waning_rate = SEIARV_1.waning_rate
        water_input = SEIARV_1.water_input
        diagnosed = SEIARV_1.diagnosed
        diagnosed_V = SEIARV_1.diagnosed_V
        vac_plan = SEIARV_1.vac_plan
        # Calculation
        N1 = S + E + A + I + D + R1 + R2
        N2 = V + E_V + A_V + I_V + D_V + R1_V + R2_V
        N_stay = N1 + N2

        ## non-vac individuals with infectiveness in University
        Infective = k_E * E + k_A * A + I
        ## vac individuals with infectiveness in University
        Infective_V = k_E * E_V + k_A * A_V + I_V
        ## non-vac infection of students in university
        Infec = beta * ctrl(t) * S * (Infective + nu * Infective_V) / N_stay
        ## vac infection of students in university
        Infec_V = epsilon * beta * ctrl(t) * V * (
                Infective + nu * Infective_V) / N_stay
        ## Viral shedding in University
        shed = p_0 * (I + D + p_A * A + p_E * E + p_R * R1) + p_1 * (
                I_V + D_V + p_A * A_V + p_E * E_V)

        ## Choice of vaccination term phi(t)S or phi(t); vac_rate=0, phi(t); vac_rate=1, phi(t) S
        vac = vac_plan(t) * (vac_rate * S + 1 - vac_rate)
        diag_A = diagnosed(t) * (diag_rate * A + 1 - diag_rate)
        diag_I = diagnosed(t) * (diag_rate * I + 1 - diag_rate)
        diag_A_V = diagnosed_V(t) * (diag_rate * A_V + 1 - diag_rate)
        diag_I_V = diagnosed_V(t) * (diag_rate * I_V + 1 - diag_rate)

        # dydt
        dS = nature_birth - Infec - vac - nature_death * S
        dE = Infec - sigma * E - nature_death * E
        dA = rho * sigma * E - gamma_A * A - nature_death * A - diag_A
        dI = (1 - rho) * sigma * E - gamma_I * I - nature_death * I - diag_I
        dD = diag_A + diag_I - gamma_D * D
        dR1 = gamma_D * D + gamma_A * A + gamma_I * I - gamma_R * R1 - nature_death * R1
        dR2 = gamma_R * R1 - nature_death * R2
        dV = vac - Infec_V - nature_death * V
        dE_V = Infec_V - sigma_V * E_V - nature_death * E_V
        dA_V = rho_V * sigma_V * E_V - gamma_A_V * A_V - nature_death * A_V - diag_A_V
        dI_V = (1 - rho_V) * sigma_V * E_V - gamma_I_V * I_V - nature_death * I_V - diag_I_V
        dD_V = diag_A_V + diag_I_V - gamma_D_V * D_V
        dR1_V = gamma_D_V * D_V + gamma_A_V * A_V + gamma_I_V * I_V - gamma_R_V * R1_V - nature_death * R1_V
        dR2_V = gamma_R_V * R1_V - nature_death * R2_V
        dW = shed / M - waning_rate(t) * W
        dM = water_input(t) - lambda_0 * M
        return np.array(
            [dS, dE, dA, dI, dD, dR1, dR2, dV, dE_V, dA_V, dI_V, dD_V, dR1_V, dR2_V, dW, dM])

    # Solving the model by odeint
    def ode_sol(self):
        init_value = [self.initvalue[i] for i in self.initvalue.keys()]
        # print("Initial Value:", init_value)
        tspan = np.arange(self.timepara["t0"], self.timepara["tend"], self.timepara["dt"])  # time span
        # print("Tspan:", tspan)
        para = tuple([self.para[i] for i in self.para.keys()])  # args
        # print("Parameters:", para)
        sol = odeint(self.SEIARV_one_model, init_value, tspan, para, )
        return {"tspan": tspan, "solution": sol}


class SEIARW():
    """
     Suspected-Exposed-Symptomatic Infected-Asymptomatic Infected
    -Recovered-Wastewater Model
    """

    def __init__(self, ctrl=None,
                 initS=100, initE=1, initA=1, initI=1, initD=0, initR1=0, initR2=0, initW=0,
                 t0=0, dt=0.1, tend=100,
                 N=100,
                 beta=0.2,
                 k_E=1 / 3, k_A=1 / 3, sigma=1 / 5.2, rho=0.3,
                 gamma_I=1 / 10, gamma_A=1 / 10, gamma_R=1 / 13,
                 detecting_A=0.1, detacting_I=0.1,
                 p_0=100, p_E=0.1, p_A=0.1, p_R=0.1,
                 water_volume=3000, waning_rate=1,
                 nature_birth=0, nature_death=0):
        self.ctrl = ctrl if ctrl is not None else lambda t: 1
        # time-dependent function; Initial value; time; parameters; others
        self.initvalue = {"initS": initS, "initE": initE, "initA": initA, "initI": initI, "initD": initD,
                          "initR1": initR1, "initR2": initR2,
                          "initW": initW}
        self.timepara = {"t0": t0, "dt": dt, "tend": tend}
        self.para = {"ctrl": self.ctrl,
                     "N": N,
                     "beta": beta,
                     "k_E": k_E, "k_A": k_A, "sigma": sigma, "rho": rho,
                     "gamma_I": gamma_I, "gamma_A": gamma_A, "gamma_R": gamma_R,
                     "detecting_A": detecting_A, "detecting_I": detacting_I,
                     "p_0": p_0, "p_E": p_E, "p_A": p_A, "p_R": p_R,
                     "nature_birth": nature_birth, "nature_death": nature_death,
                     "water-volume": water_volume, "waning_rate": waning_rate
                     }

    def BRN(self):
        R_0 = self.para["beta"] * ((1 - self.para["rho"]) / (self.para["gamma_I"] + self.para["detecting_I"])
                                   + self.para["k_A"] * self.para["rho"] / (
                                           self.para["gamma_A"] + self.para["detecting_A"])
                                   + self.para["k_E"] / self.para["sigma"])
        return R_0

    # Compartment models
    @staticmethod
    def SEIARW_model(y, t, ctrl=None, N=100,
                     beta=0.2,
                     k_E=1 / 3, k_A=1 / 3, sigma=1 / 5.2, rho=0.18,
                     gamma_I=1 / 12, gamma_A=1 / 10, gamma_R=1 / 13,
                     d_A=0.1, d_I=0.1,
                     p_0=100, p_E=0.1, p_A=0.1, p_R=0.1,
                     nature_birth=0, nature_death=0, M=3000, delta=1):
        S, E, A, I, D, R1, R2, W = y

        # time-dependent function
        # ctrl = SEIARW_1.ctrl  # input control function
        # Calculation
        ## non-vac individuals with infectiveness in University
        ctrl1 = ctrl if ctrl is not None else lambda t: 1
        Infec = beta * ctrl1(t) * S * (k_E * E + k_A * A + I) / N

        # dydt
        dS = nature_birth - Infec - nature_death * S
        dE = Infec - sigma * E - nature_death * E
        dA = rho * sigma * E - gamma_A * A - nature_death * A - d_A * A
        dI = (1 - rho) * sigma * E - gamma_I * I - nature_death * I - d_I * I
        dD = d_A * A + d_I * I
        dR1 = gamma_A * A + gamma_I * I - gamma_R * R1 - nature_death * R1
        dR2 = gamma_R * R1 - nature_death * R2
        dW = p_0 * (I + p_A * A + p_E * E + p_R * R1) / M - delta * W
        return np.array(
            [dS, dE, dA, dI, dD, dR1, dR2, dW])

    # Solving the model by odeint
    def ode_sol(self):
        init_value = [self.initvalue[i] for i in self.initvalue.keys()]
        # print("Initial Value:", init_value)
        tspan = np.arange(self.timepara["t0"], self.timepara["tend"], self.timepara["dt"])
        CRN = (self.para["beta"] * ((1 - self.para["rho"]) / (self.para["gamma_I"] + self.para["detecting_I"])
                                   + self.para["k_A"] * self.para["rho"] / (
                                           self.para["gamma_A"] + self.para["detecting_A"])
                                   + self.para["k_E"] / self.para["sigma"]))*self.ctrl(tspan)# time span
        # print("Tspan:", tspan)
        para = tuple([self.para[i] for i in self.para.keys()])  # args
        # print("Parameters:", para)
        sol = odeint(self.SEIARW_model, init_value, tspan, para, )
        Midvalue = np.zeros(len(sol[:, 4]))
        Midvalue[0]=self.initvalue["initD"]
        Midvalue[1:] = sol[:, 4][:-1]
        daily_confirmed = sol[:, 4] - Midvalue
        return {"tspan": tspan, "solution": sol, "newlyconfirmed": daily_confirmed,"crn":CRN}


if __name__ == '__main__':
    # ctrl = lambda t: 0.1 + 0.1 * np.sin(2 * np.pi * t)
    test_SEIARW = SEIARW(k_E=0.3, k_A=0.3)
    A = test_SEIARW.ode_sol()
    R_0 = test_SEIARW.BRN()
    print(R_0)
    plt.plot(A["tspan"], A["solution"][:, 1])
    plt.show()