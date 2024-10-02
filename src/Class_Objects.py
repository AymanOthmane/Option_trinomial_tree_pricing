from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import time
import sys

sys.setrecursionlimit(100000)

class Tree:
    def __init__(self, vol, rate, spot, div, divExDate, Nb_steps, matu,pricing_date):
        self.div=div
        self.divExDate=divExDate
        self.vol = vol
        self.matu = matu
        self.pricing_date=pricing_date
        self.rate = rate
        self.spot = spot
        self.steps = Nb_steps
        self.timestep = self.compute_date(matu,pricing_date)/self.steps
        self.alpha = np.exp(self.vol * np.sqrt(3 * self.timestep))
        self.root = None
        self.DF=self.discount_factor()

    def fn_alpha(self):
        return self.alpha

    def Build_Tree(self):
        Root = self.fn_build_root()
        """Construction de l'arbre pour chaque step, on commence par créer le noeud central à chaque step et ses 3 enfants"""
        for step in range(self.steps):
            next_node = Root.create_next_nodes(step)  #on génère les nouveaux noeuds
            M_Node, Up_Node, D_Node = next_node
            M_Node.connect_node(neighbor_up=Up_Node, neighbor_down=D_Node) #on connecte les trio de noeuds
            Root.connect_node(next_mid=M_Node, next_up=Up_Node, next_down=D_Node) #on connecte la root avec ses trios
            Root.node_proba(step) #calcul des proba des nouveaux noeuds, necessaire si il y a des divs
            Root.proba_next_node(step,Up_Node, M_Node, D_Node) #proba cumulee des noeuds suivants
            Root_up = Root

            """Construction de l'arbre vers le haut puis vers le bas, tant que le noeud central a un voisin au dessus alors on construit l'arbre"""
            while Root_up.get_neighbor_up() is not None: #on construit a partir du MID, ensuite on va construire l'arbre vers le haut puis vers le bas
                Root_up = Root_up.get_neighbor_up()
                MM_Node = Root_up.get_mid(Root_up.neighbor_down.next_mid, Root,step)
                if MM_Node.neighbor_up is None:
                    """On ne construit le nouveau noeud du haut que si sa proba est significative, idem pour la construction du bas """
                    if self.Pruning(Root_up,step):
                       Up_bound_Node = MM_Node.move_up()
                       Root_up.connect_node(next_up=Up_bound_Node)
                       Up_bound_Node.connect_node(neighbor_down=MM_Node)
                    Root_up.connect_node(next_up=MM_Node.neighbor_up, next_mid=MM_Node, next_down=MM_Node.neighbor_down)
                    Root_up.node_proba(step)
                    Root_up.proba_next_node(step, next_up=Root_up.next_up, next_mid=Root_up.next_mid, next_down=Root_up.next_down)

            """Si on sort de la boucle c'est que nous sommes a l'extrémité de la colonne root sans voisin haut pas de creation d'enfant"""

            Root_Down = Root
            while Root_Down.neighbor_down is not None:
                Root_Down = Root_Down.neighbor_down
                MM_Node = Root_Down.get_mid(Root_Down.neighbor_up.next_mid, Root,step)
                if MM_Node.neighbor_down is None:
                    if self.Pruning(Root_Down,step):
                        Up_bound_Node = MM_Node.move_down()
                        Root_Down.connect_node(next_down=Up_bound_Node)
                        Up_bound_Node.connect_node(neighbor_up=MM_Node)
                    Root_Down.connect_node(next_up=MM_Node.neighbor_up, next_mid=MM_Node, next_down=MM_Node.neighbor_down)
                    Root_Down.node_proba(step)
                    Root_Down.proba_next_node(step, next_up=Root_Down.next_up, next_mid=Root_Down.next_mid,next_down=Root_Down.next_down)

            Root = M_Node   #la Root devient le nouveau noeud central

    def discount_factor(self):
        DF=np.exp(-self.rate*self.timestep)
        return DF
    """Création de la Root initial et initialisation de sa proba à 1"""

    def fn_build_root(self):
        self.root = Node(self, self.spot)
        self.root.proba = 1
        return self.root
    """Méthode static afin de convertir l'écart entre 2 dates en nombre"""

    @staticmethod
    def compute_date(date,Pricing_date):
        diff_jour = (date - Pricing_date).days
        annee = float(diff_jour) / 365
        return annee
    """Methode Pruning qui renvoie True si la proba du noeud est suffisamment significative """

    def Pruning(self,node,step):
           if node.proba * node.proba_up(step)>10**(-7) or node.proba*node.proba_down(step)>10**(-7):
              return True
           else:
              return False


class Node:
    node_count = 0
    def __init__(self, tree, spot):
        self.tree = tree
        self.spot = spot
        self.next_up = None
        self.next_mid = None
        self.next_down = None
        self.neighbor_up = None
        self.neighbor_down = None
        self.p_down = None
        self.p_up = None
        self.p_mid = None
        self.proba = 0
        Node.node_count += 1
        self.price=None


    def forward(self):
        return self.spot * np.exp(self.tree.rate * self.tree.timestep)

    def variance(self):
        return self.spot ** 2 * np.exp(2 * self.tree.rate * self.tree.timestep) * (np.exp(self.tree.vol ** 2 * self.tree.timestep) - 1)

    """La methode dividende a pour objectif de détecter lors d'une step si il y a un détachement de dividende on regardant où se situe la div ex date par rapport à la step actuelle"""
    def Dividende(self,step):
        date_div = self.tree.divExDate
        pricing_date=self.tree.pricing_date
        date_div_year = self.tree.compute_date(date_div,pricing_date)
        current_step = (step) * self.tree.timestep
        next_step = (step+1) * self.tree.timestep

        if current_step < date_div_year < next_step:
            return True
        else:
            return False

    """Forward_adjust a pour objectif d'ajuster le forward lorsqu'il y a un dividende"""
    def forward_adjust(self,step):
        div = self.tree.div
        forward = self.forward()
        if self.Dividende(step):
            forward_adjust = forward - div
        else:
            forward_adjust = forward
        return forward_adjust

    def create_next_nodes(self,step):

        forward_adjust = self.forward_adjust(step)

        self.next_mid = Node(self.tree, forward_adjust)
        self.next_up = Node(self.tree, self.next_mid.spot*self.tree.alpha)
        self.next_down = Node(self.tree, self.next_mid.spot/self.tree.alpha)

        return self.next_mid, self.next_up, self.next_down

    def get_neighbor_up(self):
        return self.neighbor_up

    def get_neighbor_down(self):
        return self.neighbor_down


    """Connexion des noeuds entre eux, les arguments sont optionnels ce qui permet de réaliser tous types de connexion """
    def connect_node(self, next_up=None, next_mid=None, next_down=None,neighbor_up=None, neighbor_down=None):
        if neighbor_up is not None:
            self.neighbor_up = neighbor_up
            self.neighbor_up.neighbor_down = self
        if neighbor_down is not None:
            self.neighbor_down = neighbor_down
            self.neighbor_down.neighbor_up=self
        if next_up is not None:
            self.next_up = next_up
        if next_mid is not None:
            self.next_mid = next_mid
        if next_down is not None:
            self.next_down = next_down

    def move_up(self):
        if self.neighbor_up is None:
            return Node(self.tree, self.spot * self.tree.alpha)
        else:
            return self.neighbor_up

    def move_down(self):
        if self.neighbor_down is None:
            return Node(self.tree,self.spot/self.tree.alpha)
        else:
            return self.neighbor_down

    """Les fonctions proba_down et proba_up sont necessaires pour le pruning"""
    def proba_down(self,step):

        forward_adjust=self.forward_adjust(step)

        p_down = ((1 / (forward_adjust ** 2) * (self.variance() + forward_adjust ** 2)) - 1 - (
                              (self.tree.alpha + 1) * (
                              (1 / forward_adjust) * forward_adjust - 1))) \
                 / ((1 - self.tree.alpha) * (1 / (self.tree.alpha ** 2) - 1))
        return p_down

    def proba_up(self,step):
        p_up = self.proba_down(step) / self.tree.alpha
        return p_up

    """Cette fonction de proba itère sur les steps donc l'argument step est necessaire afin de pouvoir vérifier s'il y a un div et pour le pruning"""
    def node_proba(self,step):#calcul des probas pour les nouveaux noeuds en partant de la branche d'avant, ensuite on ajoute les probas des chemins
        forward_adjust = self.forward_adjust(step)

        """Dans la partie else, cela signifie que le noeud n'est pas significatif donc attribution des probas autrement"""
        if self.tree.Pruning(self,step):

            self.p_down= ((1 / (self.next_mid.spot ** 2) * (self.variance() + forward_adjust ** 2)) - 1 - (
                              (self.tree.alpha + 1) * (
                              (1 / self.next_mid.spot) * forward_adjust - 1))) \
                 / ((1 - self.tree.alpha) * (1 / (self.tree.alpha ** 2) - 1))

            """Changement de proba s'il y a un dividende"""

            if forward_adjust != self.forward():
                self.p_up=(((1/self.next_mid.spot)*forward_adjust-1)-((1/self.tree.alpha)-1)*self.p_down)/(self.tree.alpha-1)
            else:
                self.p_up=self.p_down/self.tree.alpha

            self.p_mid= 1 - self.p_down - self.p_up

        else:
            self.p_mid=1
            self.p_down = 0
            self.p_up = 0

    """Calcul des proba cumulées"""
    def proba_next_node(self,step,next_up=None,next_mid=None,next_down=None):
        if self.tree.Pruning(self,step):
            next_down.proba += self.proba * self.p_down
            next_mid.proba += self.proba * self.p_mid
            next_up.proba += self.proba * self.p_up

    """La fonction ci-dessous permet de détecter quel est le bon mid lors d'un détachement du div"""
    def get_mid(self, next, root,step):
        fwd=self.forward()
        forward_adjust=self.forward_adjust(step)

        if next.Is_close(forward_adjust):
            return next.neighbor_down
        elif fwd > self.spot>root.spot:
            while not next.Is_close(forward_adjust):
                next_2 = next.move_up()
                next_2.connect_node(neighbor_down=next)
                next = next_2
        else:
            while not next.Is_close(forward_adjust):
                next_2 = next.move_down()
                next_2.connect_node(neighbor_up=next)
                next = next_2
        return next

    def Is_close(self, fwd):
         if self.spot * (1 + self.tree.alpha) / 2 >= fwd and self.spot * (1 + (1 / self.tree.alpha)) / 2 <= fwd:
             return True
         else:
             return False

    """Fonction de pricing, 1) vérification de la nature de l'option grâce à opt.verif_nat, 2) pricing européen récursifs des noeuds en fonction de si on est à la dernière step ou pas"""

    def compute_pricing_node(self,opt):
        opt.verif_nat()
        if self.price is not None:
            return self.price
        elif self.next_mid is None:
            self.price=opt.payoff(self.spot)
        else: # si des noeuds ne sont pas crées à cause du pruning, le pricing diffère pour ces noeuds là
            if self.next_up==None or self.next_down==None:  #si ya du pruning en gros
                self.price = self.tree.DF * (self.p_mid * self.next_mid.compute_pricing_node(opt))
            else:
                self.price= self.tree.DF * (self.p_up * self.next_up.compute_pricing_node(opt) + self.p_down * self.next_down.compute_pricing_node(opt) + self.p_mid * self.next_mid.compute_pricing_node(opt))

        """ récupération du prix européen puis appel de la fonction pricing exotique"""

        price_euro=self.price
        self.compute_price_exo(opt,price_euro)

        return self.price

    """Fonction pricing exotique pour les options américaines, down and out et up and out"""
    def compute_price_exo(self,opt,price_euro):
        opt.verif_barrier(self.tree)
        if opt.Nat=="UO": #les noeuds supérieurs à la barrière valent 0, explication dans le word
            if self.spot>=opt.Barrier*self.tree.root.spot:
                self.price=0
        if opt.Nat=="Am":
            self.price=max(price_euro,opt.payoff(self.spot))
        if opt.Nat=="DO":
            if self.spot<=opt.Barrier*self.tree.root.spot:
                self.price=0

class Option:
    def __init__(self, strike,Type,Nat,Barrier=None):
        self.strike=strike
        self.Type=Type  #call ou put
        self.Nat=Nat #european, american, Down and out, up and out
        self.Barrier=Barrier if Barrier is not None else 0

    """si la fonction n'est pas un call, alors c'est un put"""
    def type(self):
        return self.Type=="Call"

    def payoff(self,spot):
        if self.type()==True:
            return max(spot-self.strike,0)
        else:
            return max(self.strike-spot,0)

    def verif_nat(self):
        valid_Nat = ["Am", "Eu", "UO","DO"]
        if self.Nat not in valid_Nat:
            raise OptionNatureInvalide("La valeur de 'Nat' n'est ni 'Am' ni 'Eu'")

    """ methode de vérification de la barrière"""
    def verif_barrier(self,tree):
        if self.Nat=="UO":
            if self.Barrier * tree.root.spot < tree.root.spot or self.Barrier*tree.root.spot<self.strike:
                raise BarrierInvalide("La barrier ne peut pas être inférieur au spot ou la barrière ne peut pas être inf au strike")
        elif self.Nat=="DO":
            if self.Barrier * tree.root.spot > tree.root.spot or self.Barrier*tree.root.spot>self.strike:
                raise BarrierInvalide("La barrier ne peut pas être supérieur au spot ou la barriere ne peut pas être sup au strike")

"""CLASSE GESTION ERREURS"""
class OptionNatureInvalide(Exception):
    pass

class BarrierInvalide(Exception):
    pass
