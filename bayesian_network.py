import math
from pomegranate import *

#defining conditional probability tables
A_tab = DiscreteDistribution({"ON" : 2/3, "OFF" : 1/3})
B_tab = ConditionalProbabilityTable([["ON", "ON", 0.9], ["ON", "OFF", 0.1], ["OFF", "ON", 0.1], ["OFF", "OFF", 0.9]], [A_tab])
C_tab = ConditionalProbabilityTable([['ON','ON', 0.2], ['ON','OFF',0.8], ['OFF','ON',0.8], ['OFF','OFF',0.2]], [A_tab])
table = [['ON','ON','ON',0.95], ['ON','ON','OFF',0.05], ['ON','OFF','ON',0.9], ['ON','OFF','OFF',0.1], ['OFF','ON','ON',0.3], ['OFF','ON','OFF',0.7], ['OFF','OFF','ON',0.1], ['OFF','OFF','OFF',0.9]]
D_tab = ConditionalProbabilityTable(table, [A_tab, B_tab])
E_tab = ConditionalProbabilityTable([['ON','ON', 0.2], ['ON','OFF',0.8], ['OFF','ON',0.8], ['OFF','OFF',0.2]], [D_tab])

#making nodes with respective probability table
node1 = Node(A_tab, name = "A")
node2 = Node(B_tab, name = "B")
node3 = Node(C_tab, name = "C")
node4 = Node(D_tab, name = "D")
node5 = Node(E_tab, name = "E")

#adding edges between the nodes
network = BayesianNetwork("switch")
network.add_nodes(node1,node2,node3,node4,node5)
network.add_edge(node1,node2)
network.add_edge(node1,node3)
network.add_edge(node1,node4)
network.add_edge(node2,node4)
network.add_edge(node4,node5)
network.bake() #compiling the network

#calculating P(A=ON, B=ON, C=ON, D=ON, E=ON)
query = network.predict_proba({'A':'ON', 'B': 'ON', 'C' : 'ON' , 'D' : 'ON' , 'E' : 'ON'})

print("Probability of p(E)\n", query[4])
