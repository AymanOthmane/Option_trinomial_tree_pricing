import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import networkx as nx
import sys

sys.setrecursionlimit(100000)

def black_scholes_option_price(option_type, strike, maturity, vol, r, dividends_euro, spot_price):
    dividend_yield = dividends_euro / spot_price

    d1 = (np.log(spot_price / strike) + (r - dividend_yield + 0.5 * vol**2) * maturity) / (vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)

    if option_type == 'Call':
        option_price = spot_price * np.exp(-dividend_yield * maturity) * si.norm.cdf(d1, 0.0, 1.0) - strike * np.exp(-r * maturity) * si.norm.cdf(d2, 0.0, 1.0)
        #delta = np.exp(-dividend_yield * maturity) * si.norm.cdf(d1, 0.0, 1.0)
        #vega = spot_price * np.exp(-dividend_yield * maturity) * np.sqrt(maturity) * si.norm.pdf(d1, 0.0, 1.0)/100
        #gamma = np.exp(-dividend_yield * maturity) * si.norm.pdf(d1, 0.0, 1.0) / (spot_price * vol * np.sqrt(maturity))
        # theta = df = np.exp(-r * maturity)
        #dfq = np.exp(-dividend_yield * maturity)
        #tmptheta = (1.0/365.0) * (-0.5 * spot_price * dfq * si.norm.pdf(d1) * vol / (np.sqrt(maturity)) - r * strike * df * si.norm.cdf(d2))
        #theta = dfq * tmptheta
        #rho = maturity * strike * np.exp(-r * maturity) * si.norm.cdf(d2, 0.0, 1.0)/100

    elif option_type == 'Put':
        option_price = strike * np.exp(-r * maturity) * si.norm.cdf(-d2, 0.0, 1.0) - spot_price * np.exp(-dividend_yield * maturity) * si.norm.cdf(-d1, 0.0, 1.0)
        #delta = -np.exp(-dividend_yield * maturity) * si.norm.cdf(-d1, 0.0, 1.0)
        #vega = spot_price * np.exp(-dividend_yield * maturity) * np.sqrt(maturity) * si.norm.pdf(d1, 0.0, 1.0) / 100
        #gamma = np.exp(-dividend_yield * maturity) * si.norm.pdf(d1, 0.0, 1.0) / (spot_price * vol * np.sqrt(maturity))
        #df = np.exp(-r * maturity)
        #dfq = np.exp(-dividend_yield * maturity)
        #tmptheta = (1.0 / 365.0) * (-0.5 * spot_price * dfq * si.norm.pdf(d1) * vol / (np.sqrt(maturity)) + r * strike * df * si.norm.cdf(-d2))
        #theta = dfq * tmptheta
        #rho = -maturity * strike * np.exp(-r * maturity) * si.norm.cdf(-d2, 0.0, 1.0)/100

    else:
        raise ValueError("Le type d'option doit être 'Call' ou 'Put'.")

    return option_price#, delta, vega, gamma,theta,rho

def plot_tree(self):
    G = nx.Graph()
    nodes = [(self.root, None, 0)]  # Utilisez une liste pour parcourir les nœuds de manière itérative

    while nodes:
        current, parent, depth = nodes.pop()
        G.add_node(current, pos=(current.spot, -depth))
        if parent is not None:
            G.add_edge(parent, current)

        # Ajouter les nœuds enfants à la liste pour traitement ultérieur
        if current.next_up:
            nodes.append((current.next_up, current, depth + 1))
        if current.next_mid:
            nodes.append((current.next_mid, current, depth + 1))
        if current.next_down:
            nodes.append((current.next_down, current, depth + 1))

    pos = nx.get_node_attributes(G, 'pos')
    labels = {node: f"{node.price:.3f}" for node in G.nodes()}
    #labels = {node: f"Spot: {node.spot:.2f}\nProba: {node.proba:.4f}" for node in G.nodes()}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, node_color='skyblue', font_size=8,
            font_color='black')
    plt.title("Arbre trinomial d'options")
    plt.axis('off')
    plt.show()

def convergence_BS(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Nb_step):
    differences = []
    range_step=range(2,Nb_step)
    for i in range_step:

       Nb_steps = i

       option = Option(strike, Type, Nat)
       tree = Tree(vol, rate, spot, div, divExDate, Nb_steps, matu, pricing_date)
       tree.Build_Tree()
       diff= (tree.root.compute_pricing_node(option) - black_scholes_option_price(Type, strike,tree.compute_date(matu, pricing_date),vol, rate, div, spot))*i
       differences.append(diff)
       del tree
    return plot_convergence(differences,range_step)


def plot_convergence(differences, range_step):
    plt.figure()
    plt.plot(range_step, differences)
    plt.xlabel('Nombre de pas (i)')
    plt.ylabel('Différence (Arbre - Black-Scholes)')
    plt.title('Convergence du prix de l option')
    plt.grid(True)
    plt.show()