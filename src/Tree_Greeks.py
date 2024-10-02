from datetime import timedelta
from Class_Objects import Tree, Option
import sys

sys.setrecursionlimit(100000)




def delta(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Barrier,Nb_steps):
    spot_delta_up=spot*1.01
    spot_delta_down=spot*0.99
    option = Option(strike, Type, Nat, Barrier)
    tree_up = Tree(vol, rate, spot_delta_up, div, divExDate, Nb_steps, matu, pricing_date)
    tree_up.Build_Tree()
    tree_down = Tree(vol, rate, spot_delta_down, div, divExDate, Nb_steps, matu, pricing_date)
    tree_down.Build_Tree()
    delta= (tree_up.root.compute_pricing_node(option)-tree_down.root.compute_pricing_node(option))/(spot_delta_up-spot_delta_down)
    return delta

def vega(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Nb_steps,Prix,Barrier):
    vol=vol+0.01
    option = Option(strike, Type, Nat, Barrier)
    tree = Tree(vol, rate, spot, div, divExDate, Nb_steps, matu, pricing_date)
    tree.Build_Tree()
    Prix_bump=tree.root.compute_pricing_node(option)
    del tree
    return Prix_bump - Prix

def gamma(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Barrier,Nb_steps,Delta):
    spot_delta_up_up = spot * 1.01* 1.01
    spot_delta_down_down = spot * 0.99*1.01
    option = Option(strike, Type, Nat, Barrier)
    tree_up_up = Tree(vol, rate, spot_delta_up_up, div, divExDate, Nb_steps, matu, pricing_date)
    tree_up_up.Build_Tree()
    tree_down_down = Tree(vol, rate, spot_delta_down_down, div, divExDate, Nb_steps, matu, pricing_date)
    tree_down_down.Build_Tree()
    delta_1 = (tree_up_up.root.compute_pricing_node(option) - tree_down_down.root.compute_pricing_node(option)) / (spot_delta_up_up - spot_delta_down_down)
    gamma=(delta_1-Delta)
    del tree_up_up
    del tree_down_down
    return gamma

def theta(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Barrier,Nb_steps,Prix):
    pricing_date = pricing_date + timedelta(days=1)
    option = Option(strike, Type, Nat, Barrier)
    tree = Tree(vol, rate, spot, div, divExDate, Nb_steps, matu, pricing_date)
    tree.Build_Tree()
    theta = (tree.root.compute_pricing_node(option) - Prix)
    del tree
    return theta

def rho(vol,rate,spot,div,pricing_date,matu,divExDate,strike,Type,Nat,Barrier,Nb_steps,Prix):
    rate = rate +0.01
    option = Option(strike, Type, Nat, Barrier)
    tree = Tree(vol, rate, spot, div, divExDate, Nb_steps, matu, pricing_date)
    tree.Build_Tree()
    rho = (tree.root.compute_pricing_node(option) - Prix)
    del tree
    return rho