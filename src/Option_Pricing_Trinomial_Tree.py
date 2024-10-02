from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import xlwings as xw
import time
import sys
from Class_Objects import Tree, Option
import Tree_Greeks as TG
from Others import black_scholes_option_price, plot_tree, plot_convergence

sys.setrecursionlimit(100000)


@xw.func()
def testXLwings():
    return main()

def main():
    """La barrière n'a pas d'impact si l'option n'est pas UO ou DO donc ne pas s'en préoccuper pour une option EU ou Am classique """
    start_time = time.time()
    vol = 0.25
    rate = 0.0146
    spot = 100
    div = 1.36
    pricing_date=datetime(2022,8,23)
    matu = datetime(2023,4,7)
    divExDate = datetime(2022, 12,8)
    Nb_steps = 100
    strike=101
    Type="UO"
    Nat="Eu"
    Barrier=1.1

    """PRICING"""
    option = Option(strike, Type,Nat,Barrier)
    tree = Tree(vol, rate, spot, div, divExDate, Nb_steps, matu,pricing_date)
    tree.Build_Tree()
    prix=tree.root.compute_pricing_node(option)
    print(prix)

    """CALCUL DES GRECQUES SI OPTION EU"""

    Delta=TG.delta(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Barrier,Nb_steps)
    print(f"le delta de l'option est:{Delta}")
    print(f"le vega de l'option est:{TG.vega(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Nb_steps,prix,Barrier)}")
    # print(f"le gamma de l'option est:{gamma(vol, rate, spot, div, pricing_date, matu, divExDate, strike, Type, Nat, Barrier, Nb_steps,Delta)}")
    # print(f"le theta de l'option est:{theta(vol, rate, spot, div, pricing_date, matu, divExDate, strike, Type, Nat, Barrier, Nb_steps, prix)}")
    # print(f"le rho de l'option est:{rho(vol, rate, spot, div, pricing_date, matu, divExDate, strike, Type, Nat, Barrier, Nb_steps, prix)}")

    """PRIX B&S juste pour check"""
    #print(f"le prix B&S est:{black_scholes_option_price(Type,strike,tree.compute_date(matu,pricing_date),vol,rate,div,spot)}")

    """Affichage du nombre de node crées"""
    #print(f"Nombre total d'objets Node créés : {Node.node_count}")

    """CONVERGENCE B&S"""
    #convergence_BS(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Nb_steps)

    """TEMPS D'EXEC, si on affiche le graphique, il faut quitter le graphique pour stopper le temps"""
    end_time = time.time()  # Enregistrez le temps de fin
    execution_time = end_time - start_time
    print(f"Temps d'exécution : {execution_time} secondes")

    """AFFICHAGE (PAS PLUS DE 12 STEPS)"""

    #plot_tree(tree)

    """Ci-dessous les méthodes pour calculer les grecques et faire la convergence """


main()


#if name == "main":


