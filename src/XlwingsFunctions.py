from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import xlwings as xw
import time
import sys
from Class_Objects import Tree, Option, Node
import Tree_Greeks as TG
import Others

sys.setrecursionlimit(100000)

@xw.func()
def testeXLws():
    return "works"


@xw.func()
def Build_Price_Tree(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type_C_P,Nat,Nb_steps,Barrier=None, ReturnNode=False, ReturnTimeTree= False, ReturnTimePrice = False):
    Node.node_count=0
    if ReturnTimeTree or ReturnTimePrice:
        start_time= time.time()
    Nb_steps=int(Nb_steps)
    option = Option(strike, Type_C_P, Nat, Barrier)
    tree = Tree(vol, rate, spot, div, divExDate, Nb_steps, matu, pricing_date)
    tree.Build_Tree()
    if ReturnTimeTree:
        end_time = time.time()
        return end_time-start_time
    Price = tree.root.compute_pricing_node(option)

    if ReturnTimePrice:
        end_time = time.time()
        return end_time-start_time
    elif ReturnNode:
        return Node.node_count
    else:
        return Price


@xw.func()
def TreeDelta(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type_C_P,Nat,Nb_steps, Barriere=None):
    Nb_steps=int(Nb_steps)
    return TG.delta(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type_C_P,Nat,Barriere,Nb_steps)


@xw.func()
def TreeGamma(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type_C_P,Nat,Nb_steps,Delta,Barrier=None):
    return TG.gamma(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type_C_P,Nat,Barrier,int(Nb_steps),Delta)


@xw.func()
def TreeVega(vol, rate, spot, div, pricing_date, matu, divExDate, strike, Type_C_P, Nat, Nb_steps, Price, Barrier=None):
    return TG.vega(vol, rate, spot, div, pricing_date, matu, divExDate, strike, Type_C_P, Nat, int(Nb_steps), Price, Barrier)


@xw.func()
def TreeTheta(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type_C_P,Nat,Nb_steps,Prix,Barrier=None):
    return TG.theta(vol, rate, spot, div, pricing_date, matu, divExDate, strike, Type_C_P, Nat, Barrier, int(Nb_steps), Prix)


@xw.func()
def TreeRho(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type_C_P,Nat,Nb_steps,Prix, Barrier=None):
    return TG.rho(vol, rate, spot, div, pricing_date, matu, divExDate, strike, Type_C_P, Nat, Barrier, int(Nb_steps), Prix)


@xw.func()
def black_Schole(Type_C_P,Strike,Maturity, Vol, Rate,Div,Spot):
    return Others.black_scholes_option_price(Type_C_P, Strike, Maturity, Vol, Rate, Div, Spot)


@xw.func()
def black_Schole_delta(Type_C_P,Strike,Maturity, Vol, Rate,Div,Spot):
    d1 = (np.log(Spot / Strike) + (Rate - Div + 0.5 * Vol ** 2) * Maturity) / (Vol * np.sqrt(Maturity))
    if Type_C_P=="Call":
        return np.exp(-Div * Maturity) * norm.cdf(d1, 0.0, 1.0)
    else:
        return -np.exp(-Div * Maturity) * norm.cdf(-d1, 0.0, 1.0)


@xw.func()
def black_Schole_gamma(Type_C_P,Strike,Maturity, Vol, Rate,Div,Spot):
    d1 = (np.log(Spot / Strike) + (Rate - Div + 0.5 * Vol ** 2) * Maturity) / (Vol * np.sqrt(Maturity))
    if Type_C_P == "Call":
        return np.exp(-Div * Maturity) * norm.pdf(d1, 0.0, 1.0) / (Spot * Vol * np.sqrt(Maturity))
    else:
        return np.exp(-Div * Maturity) * norm.pdf(d1, 0.0, 1.0) / (Spot * Vol * np.sqrt(Maturity))


@xw.func()
def black_Schole_vega(Type_C_P,Strike,Maturity, Vol, Rate,Div,Spot):
    d1 = (np.log(Spot / Strike) + (Rate - Div + 0.5 * Vol ** 2) * Maturity) / (Vol * np.sqrt(Maturity))
    if Type_C_P == "Call":
        return Spot * np.exp(-Div * Maturity) * np.sqrt(Maturity) * norm.pdf(d1, 0.0, 1.0)/100
    else:
        return Spot * np.exp(-Div * Maturity) * np.sqrt(Maturity) * norm.pdf(d1, 0.0, 1.0) / 100


@xw.func()
def black_Schole_Theta(vol, rate, spot, div, matu, strike, Type_C_P):

    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol ** 2) * matu) / (vol * np.sqrt(matu))
    d2 = d1 - vol * np.sqrt(matu)
    df = np.exp(-rate * matu)
    dfq = np.exp(-div * matu)

    if Type_C_P == "Call":
        tmptheta = (1.0/365.0) * (-0.5 * spot * dfq * norm.pdf(d1) * vol / (np.sqrt(matu)) - rate * strike * df * norm.cdf(d2))
        theta = dfq * tmptheta
    else:
        tmptheta = (1.0 / 365.0) * (-0.5 * spot * dfq * norm.pdf(d1) * vol / (np.sqrt(matu)) + rate * strike * df * norm.cdf(-d2))
        theta = dfq * tmptheta
    return theta


@xw.func()
def black_Schole_Rho(vol, rate, spot, div, matu, strike, Type_C_P):
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol ** 2) * matu) / (vol * np.sqrt(matu))
    d2 = d1 - vol * np.sqrt(matu)
    if Type_C_P == "Call":
        return matu * strike * np.exp(-rate * matu) * norm.cdf(d2, 0.0, 1.0) / 100
    else:
        return -matu * strike * np.exp(-rate * matu) * norm.cdf(-d2, 0.0, 1.0) / 100