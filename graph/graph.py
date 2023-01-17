from re import T
from unittest import result
import numpy as np
import ctypes

class HistoryFinder:
    def __init__(self):
        self.lib = ctypes.CDLL('./graph/graph.dll')
        #self.lib = ctypes.CDLL('D:/paper/code/dynamicgraph/VLDB/code0624/graph/graph.dll')
        self.lib.ConstructGraph.argtypes = []
        self.lib.ConstructGraph.restype = ctypes.c_void_p


        self.lib.Test.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.Test.restype = ctypes.c_void_p

        self.lib.GetInternalValue.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.GetInternalValue.restype = ctypes.c_int

        #set the node num and edge num of the graph. Pre-allocate the space.
        self.lib.InitialGraph.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.InitialGraph.restype = ctypes.c_void_p

        self.lib.GetRecordsNumBefore.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.GetRecordsNumBefore.restype = ctypes.c_void_p

        self.lib.InitNeighbors.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.lib.InitNeighbors.restype = ctypes.c_void_p

        self.lib.GetNeighborList.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.GetNeighborList.restype = ctypes.c_void_p

        self.lib.GetTSList.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.GetTSList.restype = ctypes.c_void_p

        self.lib.GetEdgeidList.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.GetEdgeidList.restype = ctypes.c_void_p

        self.lib.GetEdgeTime.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.GetEdgeTime.restype = ctypes.c_int


        self.lib.ComputeUnionAndIntersect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ComputeUnionAndIntersect.restype = ctypes.c_void_p

        self.lib.ComputeUnionAndIntersectWithEid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ComputeUnionAndIntersectWithEid.restype = ctypes.c_void_p

        self.lib.GetUnionSize.argtypes = [ctypes.c_void_p]
        self.lib.GetUnionSize.restype = ctypes.c_int

        self.lib.GetIntersectionSize.argtypes = [ctypes.c_void_p]
        self.lib.GetIntersectionSize.restype = ctypes.c_int

        self.lib.PrintNeighbors.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.PrintNeighbors.restype = ctypes.c_void_p

        self.lib.ComputeCoNeighbors.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ComputeCoNeighbors.restype = ctypes.c_int

        self.lib.GetCoNeighbors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.GetCoNeighbors.restype = ctypes.c_void_p

        self.lib.PrintNeighborWithTime.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.PrintNeighborWithTime.restype = ctypes.c_void_p


        self.lib.ComputeMaxCoNeiNum.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.ComputeMaxCoNeiNum.restype = ctypes.c_int

        self.lib.ComputeSortedCoNeighbors.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ComputeSortedCoNeighbors.restype = ctypes.c_void_p


        self.lib.GetSortedCoNeighbors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.GetSortedCoNeighbors.restype = ctypes.c_void_p

        self.lib.ComputeSmallCoNeighbors.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ComputeSmallCoNeighbors.restype = ctypes.c_int

        self.lib.ComputeSmallUnionAndIntersect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ComputeSmallUnionAndIntersect.restype = ctypes.c_void_p

        self.lib.GetDegreeBeforeEdgeid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.GetDegreeBeforeEdgeid.restype = ctypes.c_int

        self.lib.GetTimeIntervalBeforeEdgeid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.GetTimeIntervalBeforeEdgeid.restype = ctypes.c_int

        #self.lib.Debug.argtypes = [ctypes.c_void_p]
        #self.lib.Debug.restype = ctypes.c_void_p

        self.max_co_nei_num = 0;

        self.debug = False

        self.graph = self.lib.ConstructGraph()
    
        #self.lib.Debug(self.graph)

    
    #def get_edge_time(self, edge_id):
    #    return self.lib.GetEdgeTime(self.graph, edge_id)

    def print_neighbor_with_time(self, node, time):
        self.lib.PrintNeighborWithTime(self.graph, node, time)

    def print_neighbor(self, node):
        self.lib.PrintNeighbors(self.graph, node)

    def find_before_start(self):
        return self.lib.GetInternalValue(self.graph, 0)

    def find_before_end(self):
        return self.lib.GetInternalValue(self.graph, 1)

    def find_before_total_num(self):
        return self.lib.GetInternalValue(self.graph, 2)

    def get_degree_before_edge(self, src, edge_id):
        if edge_id is None:
            print('edge_id is None')
            return
        return self.lib.GetDegreeBeforeEdgeid(self.graph, src, edge_id)

    def get_time_interval_before_edge(self, src, edge_id, num):
        if edge_id is None:
            print('edge_id is None')
            return
        return self.lib.GetTimeIntervalBeforeEdgeid(self.graph, src, edge_id, num)

    
    def init_off_set(self, adj_list):
        node_num = len(adj_list)
        edge_num = 0
        for i in range(len(adj_list)):
            edge_num += len(adj_list[i])
        self.lib.InitialGraph(self.graph, node_num, edge_num);
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x : x[2])
            #print(curr)
            degree = len(curr)
            neighbors = np.array([x[0] for x in curr], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            timestamps = np.array([int(x[2]) for x in curr], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            eids = np.array([x[1] for x in curr], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

            self.lib.InitNeighbors(self.graph, i, degree, neighbors, timestamps, eids)
        
        #self.max_co_nei_num = self.lib.ComputeMaxCoNeiNum(self.graph, 50)
       

    def get_co_neighbors(self, node1, node2, time):
        num = self.lib.ComputeCoNeighbors(self.graph, node1, node2, time)
        if num == 0:
            return []
        else:
            nodes_ptr = (ctypes.c_int * num)()
            self.lib.GetCoNeighbors(self.graph, nodes_ptr)
            nodes_np = np.ctypeslib.as_array(nodes_ptr).tolist()
            return nodes_np

    def get_sorted_co_neighbors(self, node1, node2, time, num):
        self.lib.ComputeSortedCoNeighbors(self.graph, node1, node2, time, num)
        nodes_ptr = (ctypes.c_int * num)()
        self.lib.GetSortedCoNeighbors(self.graph, nodes_ptr)
        nodes_np = np.ctypeslib.as_array(nodes_ptr).tolist()
        return nodes_np

    def get_small_co_neighbors(self, node1, node2, time, per_num):
        num = self.lib.ComputeSmallCoNeighbors(self.graph, node1, node2, time, per_num)
        #print(num, per_num)
        if num == 0:
            return []
        else:
            nodes_ptr = (ctypes.c_int * num)()
            self.lib.GetCoNeighbors(self.graph, nodes_ptr)
            nodes_np = np.ctypeslib.as_array(nodes_ptr).tolist()
            return nodes_np


    def compute_union_and_intersection(self, node1, node2, time):
        self.lib.ComputeUnionAndIntersect(self.graph, node1, node2, time)
        union_size = self.lib.GetUnionSize(self.graph)
        intersection_size = self.lib.GetIntersectionSize(self.graph)
        return union_size, intersection_size

    def compute_small_union_and_intersection(self, node1, node2, time, per_num):
        self.lib.ComputeSmallUnionAndIntersect(self.graph, node1, node2, time, per_num)
        union_size = self.lib.GetUnionSize(self.graph)
        intersection_size = self.lib.GetIntersectionSize(self.graph)
        return union_size, intersection_size

    def compute_union_and_intersection_with_eid(self, node1, node2, eid):
        self.lib.ComputeUnionAndIntersectWithEid(self.graph, node1, node2, eid)
        union_size = self.lib.GetUnionSize(self.graph)
        intersection_size = self.lib.GetIntersectionSize(self.graph)
        return union_size, intersection_size
    
    def GetDegreeAndInterval(self, src_idx, cut_time, e_idx=None, node_num=None):
        edge_id = e_idx
        if edge_id is None:
            edge_id = -1

        self.lib.GetRecordsNumBefore(self.graph, src_idx, cut_time, edge_id)
        start = self.find_before_start()
        num = self.find_before_total_num()
        degree = num
        if node_num is not None:
            num = node_num
        time1 = self.get_edge_time(start)
        time2 = self.get_edge_time(start+num)
        interval = time2 - time1
        return degree, interval

    def find_before(self, src_idx, cut_time, e_idx=None, return_binary_prob=False, node_num=None, return_time=False):
        edge_id = e_idx
        if edge_id is None:
            edge_id = -1
        self.lib.GetRecordsNumBefore(self.graph, src_idx, cut_time, edge_id)
        start = self.find_before_start()
        end = self.find_before_end()
        num = self.find_before_total_num()
        real_degree = num
        if self.debug:
            print('[find_before]')
            print('time: ', cut_time)
            print('start', start, 'end', end, 'num', num)
            print('start', start, 'end', end, 'num', num)
            print(start, end, num)
            #self.print_neighbor(src_idx)
        if node_num is not None:
            num = node_num
        time_interval = None
        if return_time:
            time1 = self.get_edge_time(start)
            time2 = self.get_edge_time(start + num)
            time_interval = time2 - time1
        if start == end:
            if return_time:
                return [0] * num, [0] * num, [0] * num, real_degree, time_interval
            else:
                return [0] * num, [0] * num, [0] * num
        else:
            tar_list = (ctypes.c_int * num)()
            edgeid_list = (ctypes.c_int * num)()
            ts_list = (ctypes.c_int * num)()
            self.lib.GetNeighborList(self.graph, start, end, num, tar_list)
            self.lib.GetTSList(self.graph, start, end, num, ts_list)
            self.lib.GetEdgeidList(self.graph, start, end, num, edgeid_list)
            if return_time:
                return np.ctypeslib.as_array(tar_list),  np.ctypeslib.as_array(edgeid_list), np.ctypeslib.as_array(ts_list), real_degree, time_interval
            else:
                return np.ctypeslib.as_array(tar_list),  np.ctypeslib.as_array(edgeid_list), np.ctypeslib.as_array(ts_list)

    #def get_node_nb(self, src_idx_l, cut_time_l, num_neighbors, step, order, e_idx_l=None):
    def extract_neighbors(self, node_idx_l, time_l, e_idx_l=None, neighbor_num=None, need_time=False):
        node_neighbors = []
        node_eidx = []
        node_t = []
        #print('extract_neighbors')
        #print('node_idx_l', node_idx_l)
        #print(e_idx_l)
        if self.debug and e_idx_l is None:
            print('e_idx_l = None')
        if e_idx_l is None:
            for i, (src_idx, time) in enumerate(zip(node_idx_l, time_l)):
                #print('find_before: e_idx_l = None')
                #print('src', src_idx, 'time', time)
                #self.print_neighbor(src_idx)
                if need_time:
                    neighbor_idx, neighbor_eidx, neighbor_ts, degree, interval = self.find_before(src_idx, time, node_num = neighbor_num, return_time = need_time)
                else:
                    neighbor_idx, neighbor_eidx, neighbor_ts = self.find_before(src_idx, time, node_num = neighbor_num, return_time = need_time)
                #print(neighbor_idx)
                node_neighbors.append(neighbor_idx)
                node_eidx.append(neighbor_eidx)
                node_t.append(neighbor_ts)
        else:
            for i, (src_idx, e_idx, time) in enumerate(zip(node_idx_l, e_idx_l, time_l)):
                if self.debug:
                    print("find_before: e_idx_l != None")
                    print('find_before')
                    print('src_idx', src_idx)
                    print(e_idx)
                    #self.print_neighbor(src_idx)
                if need_time:
                    neighbor_idx, neighbor_eidx, neighbor_ts, degree, interval = self.find_before(src_idx, time, e_idx = e_idx, node_num= neighbor_num, return_time = need_time)
                else:
                    neighbor_idx, neighbor_eidx, neighbor_ts = self.find_before(src_idx, time, e_idx = e_idx, node_num= neighbor_num, return_time = need_time)
                if self.debug:
                    print('ngh_idx', neighbor_idx)
                node_neighbors.append(neighbor_idx)
                node_eidx.append(neighbor_eidx)
                node_t.append(neighbor_ts)
        if need_time:
            return node_neighbors, node_eidx, node_t, degree, interval
        else:
            return node_neighbors, node_eidx, node_t
        

    def get_interaction(self, src_idx_l, tag_idx_l, cut_time_l, num_neighbors, step, e_idx_l = None):
        #print('src_idx_l', src_idx_l)
        neighbor_n, neighbor_eid, neighbor_ts = self.extract_neighbors(src_idx_l, cut_time_l, e_idx_l=e_idx_l, neighbor_num=num_neighbors[0]+step)
        #print('neighbor_n', neighbor_n)
        batch = len(src_idx_l)
        partner_n, partner_eid, partner_ts = [], [], []
        for idj in range(batch):
            cut_time = [cut_time_l[idj]] * step
            #print('neighbor_n[]', neighbor_n[idj][-step:])
            #print(neighbor_n[idj][-step:], cut_time, num_neighbors[1])
            #input()
            x, y, z = self.extract_neighbors(neighbor_n[idj][-step:], cut_time, neighbor_num=num_neighbors[1], need_time=False)
            #print(x, y, z)
            #exit(0)
            partner_n.append(x)
            partner_eid.append(y)
            partner_ts.append(z)
        
        node_records = []
        eidx_records = []
        t_records = []
        node_sequence = []
        self.debug = False
        for idj in range(batch):
            node_batchj, e_idx_batchj = partner_n[idj], partner_eid[idj]
            neighbors_batchj = []
            e_batchj = []
            t_batchj = []
            nodes = []
            for n2, e2 in zip(node_batchj, e_idx_batchj):
                t2 = [cut_time_l[idj]] * len(n2)
                #print('t2', t2)
                n_l2, e_l2, t_l2 = self.extract_neighbors(n2, t2, e_idx_l=e2, neighbor_num=num_neighbors[0])
                neighbors_batchj.append(n_l2)
                e_batchj.append(e_l2)
                t_batchj.append(t_l2)
                x = np.array(n_l2)
                nodes += list(x.flatten())
            node_sequence.append(nodes)

            node_records.append(neighbors_batchj)
            eidx_records.append(e_batchj)
            t_records.append(t_batchj)
        return (neighbor_n, neighbor_eid, neighbor_ts), (partner_n, partner_eid, partner_ts), (node_records, eidx_records, t_records)



        