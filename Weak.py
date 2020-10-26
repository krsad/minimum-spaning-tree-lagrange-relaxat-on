import csv
import time
import datetime

a = datetime.datetime.now()
class Graph    :

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        self.totol_cost=0
        self.changed_edges =[]
        self.counter_for_unchanged=0

        # to store graph

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
        #print(self.graph)
        # A utility function to find set of an element i

    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

        # A function that does union of two sets of x and y

    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

            # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's
    # algorithm
    def KruskalMST(self, graph, flag_for_upper_bound=0):
        graph.cost_array_kruskal_for_upper_bound = []
        graph.edge_array_kruskal_for_upper_bound = []
        graph.edge_array_kruskal = []
        graph.cost_array_kruskal = []

        result = []  # This will store the resultant MST

        i = 0  # An index variable, used for sorted edges
        e = 0  # An index variable, used for result[]

        # Step 1:  Sort all the edges in non-decreasing
        # order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = [];
        rank = []
        counter = 0
        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
            # Number of edges to be taken is equal to V-1
        #print("graph",self.graph)
        while e < self.V-1 :
            if(flag_for_upper_bound==1):

                # Step 2: Pick the smallest edge and increment
                # the index for next iteration
                #print(i)
                #print("unsorted graph",self.graph)
                #print("graph",self.graph[i][2])
                try:
                    u, v, w = self.graph[i]
                except:
                    return -1*graph.upper_bound
                if(counter >= self.counter_for_unchanged ):
                    for k in range(len(graph.conflic_arr_just_true[0])):
                        arr = decode_edge(graph.conflic_arr_just_true[0][k])
                        if(int(arr[0])==int(u) and int(arr[1])==int(v)):

                            arr2 = decode_edge(graph.conflic_arr_just_true[1][k])
                            for y in range(len(self.graph)):
                                if(int(self.graph[y][0])==int(arr2[0]) and int(self.graph[y][1])==int(arr2[1]) and graph.conflic_arr_just_true[1][k] not in self.changed_edges):
                                    self.changed_edges.append(graph.conflic_arr_just_true[1][k])
                                    self.graph[y][2] +=1000000
                                    #print("changed graph",self.graph[y])
                                    self.graph = sorted(self.graph, key=lambda item: item[2])
                #print("sorted graph",self.graph)

                counter +=1



                #print("i",i)
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)

                # If including this edge does't cause cycle,
                # include it in result and increment the index
                # of result for next edge
                if x != y:

                    #print("x",u)
                    #print("y",v)

                    e = e + 1

                    result.append([u, v, w])
                    self.union(parent, rank, x, y)
                    #print("result", result)
                    counter+=1
            else:

                u, v, w = self.graph[i]
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)

                # If including this edge does't cause cycle,
                # include it in result and increment the index
                # of result for next edge
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.union(parent, rank, x, y)
                    # Else discard the edge

            # Else discard the edge
                #else:
                # if(counter!=self.counter_for_unchanged):

        # print the contents of result[] to display the built MST

        #print("Following are the edges in the constructed MST")
        total_cost=0


        for u, v, weight in result:
            #print str(u) + " -- " + str(v) + " == " + str(weight)
            print("%d -- %d == %d" % (u, v, weight))

            last_to_string = str(u) + '0' + str(v)
            if (flag_for_upper_bound == 0):
                graph.cost_array_kruskal.append(weight)
                graph.edge_array_kruskal.append(last_to_string)



            elif (flag_for_upper_bound == 1):
                graph.cost_array_kruskal_for_upper_bound.append(weight)
                graph.edge_array_kruskal_for_upper_bound.append(last_to_string)

            total_cost += weight

        #print(total_cost)

        return total_cost

class Subgradient:

    def __init__(self):
        self.conflict_size = 0
        self.graph_size = 0
        self.upper_bound = 0
        self.lower_bound = 0
        self.pi = 2
        self.best_lower_bound = 0
        self.best_upper_bound = 0
        self.best_upper_bound_array_edges=[]
        self.best_lower_bound_array = []
        self.best_upper_bound_aray = []
        self.a = 0
        self.edge_array_kruskal_for_upper_bound=[]
        self.edge_array_kruskal=[]
        self.cost_array_kruskal_for_upper_bound=[]
        self.cost_array_kruskal=[]
        #self.compare= 0.0
        self.conflic_arr_just_true = [[], []]
        self.total_edge_array=[]
        self.total_edge_array_not_changed=[]
        self.total_cost_array=[]
        self.total_cost_array_not_changed=[]
        self.total_conflict_edges_combined = []
        #def cbar
        self.sum_g=0
        self.cbar = []
        self.t = 1.0
        self.lambda_e = []
        self.conflict_arr = []
        self.I = 0
        self.graph_example_array = []

    def make(self):

        for i in range(self.graph_size):
            self.cbar.append([])
            for j in range(self.graph_size):
                self.cbar[i].append(0.0)


        #lambda_e 15x15 matrix
        for i in range(self.graph_size):
            self.lambda_e.append([])
            for j in range(self.graph_size):
                self.lambda_e[i].append(0.0)

        #conflict matris 38x38 matris
        for i in range(self.conflict_size):
            self.conflict_arr.append([])
            for j in range(self.conflict_size):
                self.conflict_arr[i].append(0)


        #graph_example = first cost matris 15x15
        for i in range(self.graph_size):
            self.graph_example_array.append([])
            for j in range(self.graph_size):
                self.graph_example_array[i].append(0)

class Edges :
    def __init__(self, edge):
        self.edge_id = edge
        self.numofconf = 0
        self.lambdaofconf = 0
        self.lambdavalue = 0
        self.confsumx = 0
        self.ge = 0
        self.conf = []

def decode_edge(edge):
    arr = ["", ""]
    if(len(edge)%2 ==1):
        temp = len(edge)/2-0.5
        temp = int(temp)
        #print(temp)

        arr[0] = edge[0:int(temp)]
        arr[1] = edge[int(temp+1):]

    else:
        temp = len(edge)/2
        temp = temp - 1
        while (True):
            if (edge[int(temp) + 1] != '0'):
                break
            else:
                temp += 1

        if(edge[int(temp):int(temp+1)] == '0'):
            arr[0] = edge[0:int(temp)]
            arr[1] = edge[int(temp + 1):]
        else:
            temp = temp -1
            while(True):
                if(edge[temp+1]!='0'):
                    break
                else:
                    temp += 1
            arr[0] = edge[0:int(temp)]
            arr[1] = edge[int(temp + 1):]
    return arr

def max_func(var1,var2):
    if(var1 < var2):
        return var2
    else:
        return var1


## here the file names, you can easily change and adopt to code,
## don't forget to look to deliminator it is ',' or ';' for each file
sub = Subgradient()
##this part is about auto find the file and conflict size
graph_csv = "Cost.csv"
conflict_csv = "Conf.csv"
sub.graph_size = 0
with open(graph_csv, 'r') as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    for row in readCSV:
        sub.graph_size+=1
number_of_conflict = 0
with open(conflict_csv, 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        number_of_conflict+=1
sub.conflict_size = number_of_conflict

sub.make()

g = Graph(sub.graph_size)

### to get full edges from GraphExample

j=0
i=0
with open(graph_csv, 'r') as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    for row in readCSV:
        for index in row:
            # sprint (index)

            #print(index,i,j)
            try:
                if(int(index)!=-1 and int(index)!=0 and i<j):
                    sub.graph_example_array[i][j]= int(index)
            except:
                pass
            j = j + 1
        j = 0
        i = i + 1

z = 0
for  column in range(len(sub.graph_example_array)):
    for row in range(len(sub.graph_example_array[0])):
        if (int(sub.graph_example_array[column][row])!=-1 and int(sub.graph_example_array[column][row])!=0 and column<row):
            sub.total_edge_array.append(str(column)+'0'+str(row))
            sub.total_cost_array.append(sub.graph_example_array[column][row])
            z+=1

######################################################################

## to calculate the upperbound ##

total_edge_array_copy_ = []
total_cost_array_copy_ = []
total_cost_array_copy_1_ = []

for i in sub.total_edge_array:
    total_edge_array_copy_.append(i)

for i in sub.total_cost_array:
    total_cost_array_copy_.append(i)

for i in total_cost_array_copy_:
    total_cost_array_copy_1_.append(i)

##to get conflict matris
row_iterator = 0
columnIterator = 0

with open(conflict_csv, 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        for index in row:

            sub.conflict_arr[columnIterator][row_iterator] = index
            #print(conflict_array[columnIterator][row_iterator])

            row_iterator = row_iterator + 1
        row_iterator = 0
        columnIterator = columnIterator + 1


for column in range(len(sub.conflict_arr)):
    for row in range(len(sub.conflict_arr[column])):
        if(sub.conflict_arr[column][row]=='1' and (row!=0 and column!=0)):# and row<column
            if(int(sub.conflict_arr[0][row])<102):
                sub.conflic_arr_just_true[0].append('00'+str(sub.conflict_arr[0][row]))
                if(('00'+str(sub.conflict_arr[0][row])) not in sub.total_conflict_edges_combined):
                    sub.total_conflict_edges_combined.append('00'+str(sub.conflict_arr[0][row]))
            else:
                sub.conflic_arr_just_true[0].append(sub.conflict_arr[0][row])
                if(str(sub.conflict_arr[0][row]) not in sub.total_conflict_edges_combined):
                    sub.total_conflict_edges_combined.append(sub.conflict_arr[0][row])
            if(int(sub.conflict_arr[column][0])<102):
                sub.conflic_arr_just_true[1].append('00'+str(sub.conflict_arr[column][0]))
                if(('00'+str(sub.conflict_arr[column][0])) not in sub.total_conflict_edges_combined):
                    sub.total_conflict_edges_combined.append('00'+str(sub.conflict_arr[column][0]))
            else:
                sub.conflic_arr_just_true[1].append(sub.conflict_arr[column][0])
                if((sub.conflict_arr[column][0]) not in sub.total_conflict_edges_combined):
                    sub.total_conflict_edges_combined.append(sub.conflict_arr[column][0])

k =0

##labda_ef is declined ##
edge_objects = []
indice = 0
temp_for_ef = 0
conf_arr_once = []

for edge in sub.conflic_arr_just_true[0]:
    if(edge not in conf_arr_once):
        conf_arr_once.append(edge)
        edge_objects.append(Edges(edge))
        indice += 1

for z in range(len(conf_arr_once)):
    for i in range(len(sub.conflic_arr_just_true[0])):
        if(edge_objects[z].edge_id==sub.conflic_arr_just_true[0][i]):
            for j in range(len(conf_arr_once)):
                if(edge_objects[j].edge_id == sub.conflic_arr_just_true[1][i]):
                    edge_objects[z].conf.append(j)

## equalize the cost
for cost in range(len(sub.total_cost_array_not_changed)):
    sub.total_cost_array_not_changed[cost] = -1 * sub.total_cost_array_not_changed[cost]

g_for_upper_bound = Graph(sub.graph_size)


for column in range(len(sub.graph_example_array)):
    for row in range(len(sub.graph_example_array[column])):
        if (column < row and sub.graph_example_array[column][row] != 0 and sub.graph_example_array[column][row]!=-1):
            g_for_upper_bound.addEdge(column, row, -1*sub.graph_example_array[column][row])

sum_of_all_lambda= 0
sum_of_kruskal_lambda = 0

while(True):   #until too many iterations
    sub.total_cost_array = []
    total_edge_array_copy = []
    total_cost_array_copy = []
    total_cost_array_copy_1 = []
    for i in total_edge_array_copy_:
        total_edge_array_copy.append(i)

    for i in total_cost_array_copy_:
        total_cost_array_copy.append(i)

    for i in total_cost_array_copy_1_:
        total_cost_array_copy_1.append(i)

    for i in total_cost_array_copy_1:
        sub.total_cost_array.append(i)

    ##calculate max cost##
    maximum_cost = 0
    for i in total_cost_array_copy_:
        if(i>maximum_cost):
            maximum_cost = i

    ## increament general k  ##
    k=k+1

#COST GÜNCELLEME SATIRI
    print("ilk :", sub.total_cost_array)
    for edge in range(len(sub.total_edge_array)):
        if(sub.total_edge_array[edge] in conf_arr_once):
            line = conf_arr_once.index(sub.total_edge_array[edge])
            sub.total_cost_array[edge] += (edge_objects[line].numofconf*edge_objects[line].lambdavalue)+edge_objects[line].lambdaofconf


    print("son :", sub.total_cost_array)
    #print(sub.graph_example_array)

    g = Graph(sub.graph_size)
    ## solve kruskal with updated cost ##
    i = 0
    for edge in sub.total_edge_array:
        arr = decode_edge(edge)

        sub.graph_example_array[int(arr[0])][int(arr[1])] = sub.total_cost_array[i]
        sub.graph_example_array[int(arr[1])][int(arr[0])] = sub.total_cost_array[i]
        i+=1

    for column in range(len(sub.graph_example_array)):
        for row in range(len(sub.graph_example_array[column])):
            if(column<row and sub.graph_example_array[column][row] != 0):
                g.addEdge(column, row,sub.graph_example_array[column][row])


    z1=g.KruskalMST(sub)
    z = z1 - sum_of_all_lambda
    sub.sum_g = 0
    flag_for_g = 0

    #Sum G^2 Bulma
    totalconflicts=0;
    while (totalconflicts+1 <= len(sub.conflic_arr_just_true[0])):
        conflictnumber=0;
        g2=sub.conflic_arr_just_true[1][totalconflicts]
        g2_copy=sub.conflic_arr_just_true[1][totalconflicts]
        if(g2 in sub.edge_array_kruskal):
            g2_value = 1
        else:
            g2_value = 0
        g1_value_sum=0
        while(g2==g2_copy):
            g1=sub.conflic_arr_just_true[0][totalconflicts]  ##it can be more than 2 conflict solve later
            conflictnumber +=1
            totalconflicts +=1
            if(totalconflicts+1 >= len(sub.conflic_arr_just_true[0])):
                break;
            if(g1 in sub.edge_array_kruskal):  #kruskal
                g1_value=1
            else:
                g1_value=0
            g1_value_sum +=g1_value
            g2_copy = sub.conflic_arr_just_true[1][totalconflicts]
            if((1-g1_value-g2_value)==-1):
                flag_for_g = 1
        for i in range(len(conf_arr_once)):
            if(edge_objects[i].edge_id == g2):
                edge_objects[i].numofconf = conflictnumber
                edge_objects[i].confsumx = g1_value_sum
                edge_objects[i].ge = -(conflictnumber-g1_value_sum-(conflictnumber*g2_value))

    for i in range(len(conf_arr_once)):
        sub.sum_g += edge_objects[i].ge * edge_objects[i].ge
    #Conflictsiz Sonuç Varsa Durdur
    if(flag_for_g == 0):
        print("no conflict to edit")
        totall = 0
        print("best upper bounds : ", sub.best_upper_bound_aray)
        print("best lower bounds : ", sub.best_lower_bound_array)
        print("best upper bounds edges : ", sub.best_upper_bound_array_edges)

        print("z : "+str(z))
        for i in range(len(sub.edge_array_kruskal)):
            print("edge : "+str(sub.edge_array_kruskal[i]) + " cost: "+str(sub.cost_array_kruskal[i]))

        for i in range(len(sub.edge_array_kruskal)):
            if(sub.edge_array_kruskal[i] in total_edge_array_copy_):
                line = total_edge_array_copy_.index(sub.edge_array_kruskal[i])
                print("edge : "+str(sub.edge_array_kruskal[i]) + " cost: ",total_cost_array_copy_[line])
                totall+=total_cost_array_copy_[line]
        print("total: ",totall)
        b = datetime.datetime.now()  # stop time
        print("Running Time Interval :", b - a)
        break
    else:
        print("z : "+str(z))

    ## line 15 after else ##
    sub.lower_bound = z

    if(z>sub.best_lower_bound):
        sub.best_lower_bound = z
        sub.best_lower_bound_array.append(round(z, 2))
        sub.t = 0

    else:
        sub.t += 1

    if(sub.t >=20):
        sub.t = 0
        sub.pi /=2


    ##########################################################
    ## to update the upper bound ##
    ###########################################################

    conflict_counter_for_kruskal_edges=[]
    conflict_counter_for_kruskal_edges_for_upper_bound=[]
    for i in range(len(sub.edge_array_kruskal)):
        conflict_counter_for_kruskal_edges.append(0)
        conflict_counter_for_kruskal_edges_for_upper_bound.append(0)

    for edge in range(len(sub.edge_array_kruskal)):
        for conflict in range(len(sub.conflic_arr_just_true[0])):
            if(sub.edge_array_kruskal[edge] == sub.conflic_arr_just_true[0][conflict]):
                if(sub.conflic_arr_just_true[1][conflict] in sub.edge_array_kruskal):
                    conflict_counter_for_kruskal_edges[edge] +=1

    ## find the minimum conflicted edge on kruskal ##
    # find max conflicted number on kruskal #
    max = 0
    for i in conflict_counter_for_kruskal_edges :
        if (i>max):
            max = i

    g_for_upper_bound = Graph(sub.graph_size)

    for i in range(max+1) :
        for j in range(len(conflict_counter_for_kruskal_edges)):

            if(i == conflict_counter_for_kruskal_edges[j] and sub.edge_array_kruskal[j] not in g_for_upper_bound.changed_edges):
                if(i == 0):
                    temp =total_edge_array_copy.index(sub.edge_array_kruskal[j])
                    total_cost_array_copy[temp] -=1100000
                else:
                    for k in range(len(sub.conflic_arr_just_true[0])):
                        if(sub.conflic_arr_just_true[0][k]==sub.edge_array_kruskal[j]):
                            if (sub.conflic_arr_just_true[1][k] not in g_for_upper_bound.changed_edges):
                                g_for_upper_bound.changed_edges.append(sub.conflic_arr_just_true[1][k])
                                temp1 = total_edge_array_copy.index(sub.conflic_arr_just_true[1][k])
                                total_cost_array_copy[temp1] = 1100000-total_cost_array_copy[temp1]
                                temp2 = total_edge_array_copy.index(sub.conflic_arr_just_true[0][k])
                                if(total_cost_array_copy[temp2]>-800000):
                                    total_cost_array_copy[temp2] -= 1100000
                                g_for_upper_bound.counter_for_unchanged+=1

    for cost in range(len(total_cost_array_copy)):
        temp = decode_edge(total_edge_array_copy[cost])
        if(total_cost_array_copy[cost] < 0):
            g_for_upper_bound.addEdge(int(temp[0]), int(temp[1]), int(total_cost_array_copy[cost]))
        else:
            if(total_cost_array_copy[cost]<800000):
                g_for_upper_bound.addEdge(int(temp[0]), int(temp[1]), -1*int(total_cost_array_copy[cost]))
            else:
                g_for_upper_bound.addEdge(int(temp[0]), int(temp[1]), int(total_cost_array_copy[cost]))

    edge_array = []
    edge_array = sub.edge_array_kruskal
    upperBoundTemp = g_for_upper_bound.KruskalMST(sub, flag_for_upper_bound=1)


    for edge in range(len(sub.edge_array_kruskal_for_upper_bound)):
        for conflict in range(len(sub.conflic_arr_just_true[0])):
            if (sub.edge_array_kruskal_for_upper_bound[edge] == sub.conflic_arr_just_true[0][conflict]):
                if (sub.conflic_arr_just_true[1][conflict] in sub.edge_array_kruskal_for_upper_bound):
                    conflict_counter_for_kruskal_edges_for_upper_bound[edge] += 1
    print("upperboundconflict :", conflict_counter_for_kruskal_edges_for_upper_bound)
    upper = 0

    #Upper Bound Bulma
    for i in sub.edge_array_kruskal_for_upper_bound:
        temp = total_edge_array_copy.index(i)
            # print(total_edge_array_copy[temp])
            # print(total_cost_array_copy_1[temp])
        upper += total_cost_array_copy_1[temp]
    g_for_upper_bound.changed_edges = []
    sub.upper_bound = upper
    if (sub.best_upper_bound == 0):
        sub.best_upper_bound = sub.graph_size * maximum_cost
    # Best Upper Bound Güncelleme
    if (sum(conflict_counter_for_kruskal_edges_for_upper_bound) == 0):
        if(sub.best_upper_bound>=sub.upper_bound):
            sub.best_upper_bound=sub.upper_bound
            sub.best_upper_bound_array_edges.append(sub.edge_array_kruskal_for_upper_bound)
            sub.best_upper_bound_aray.append(upper)

    print("upperbound : ", sub.upper_bound )
    print("best upperbound : ", sub.best_upper_bound )
    print("lowerbound : ", sub.lower_bound )
    print("best lower bound : ", sub.best_lower_bound)

    if(sub.best_upper_bound<sub.lower_bound):
        print("upper is bigger than lower")
        print("z : " + str(z1))
        #for i in range(len(sub.edge_array_kruskal)):
        #    print("edge : " + str(sub.edge_array_kruskal[i]) + " cost: " + str(sub.cost_array_kruskal[i]))

    ## determine the step lenght ##
    sub.a = sub.pi * (sub.best_upper_bound - sub.lower_bound) / sub.sum_g

    sum_of_all_lambda = 0
    for i in range(len(conf_arr_once)):
        edge_objects[i].lambdavalue = max_func(0, edge_objects[i].lambdavalue + (edge_objects[i].ge * sub.a))
        sum_of_all_lambda += edge_objects[i].lambdavalue*edge_objects[i].numofconf

    for i in range(len(conf_arr_once)):
        edge_objects[i].lambdaofconf=0
        for j in range(len(edge_objects[i].conf)):
            edge_objects[i].lambdaofconf += edge_objects[edge_objects[i].conf[j]].lambdavalue

        print("edge:", edge_objects[i].edge_id)
        print("lambda values:", edge_objects[i].lambdavalue)
        #print("conf number", edge_objects[i].numofconf)
        #print("sumconflamda", edge_objects[i].lambdaofconf)
        #print("sum g:" , sub.sum_g)
    print("sum_of_all_lambda", sum_of_all_lambda)

    print("stepsize: ", sub.a)
    print("ITERATION DONEEEEE")



