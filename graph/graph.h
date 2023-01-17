#pragma once
#include<vector>
#include<algorithm>
#include<cassert>
#include<set>
#include<climits>
#include<unordered_map>
#include<iostream>
#include<cassert>
using namespace std;

struct pair_hash {
	template<class T1, class T2>
	std::size_t operator() (const pair<T1, T2>& p)const {
		auto h1 = std::hash<T1>{}(p.first);
		auto h2 = std::hash<T2>{}(p.second);
		return h1 ^ h2;
	}
};

class Graph {
public:
	//edges 
	vector<int> offset;
	vector<int> targets;
	vector<int> timestamps;
	vector<int> edge_idxs;
	vector<char> flag;
	vector<set<int> > neighbor_sets;
	unordered_map<pair<int, int>, int, pair_hash> timestamp2offset;
	unordered_map<pair<int, int>, int, pair_hash> edgeid2offset;

	int union_size;
	int intersect_size;

	list<int> co_neighbors;

	int max_co_neighbor_num;

	int source_num;
	int target_num;


	//graph property
	int node_num;
	int edge_num;

	//[0] start_idx for range query, 
	//[1] end_idx for range query,
	//[2] num of before_records
	int values[2];
	bool debug;


public:
	int BinarySearch(int start, int end, int value);
	void GetNeighborList(int start, int end, int num, int* tar_list);
	void GetTSList(int start, int end, int num, int* ts_list);
	void GetEdgeidList(int start, int end, int num, int* edgeid_list);
	void RecordsBeforeNum(int src, int time, int cut_edge_idx);
	void ComputeUnionAndIntersect(int node1, int node2, int time);
	void ComputeUnionAndIntersectWithEid(int node1, int node2, int eid);
	void ComputeSmallUnionAndIntersect(int node1, int node2, int time, int num);
	void ComputeCoNeighbors(int node1, int node2, int time);
	void ComputeTimeSortedCoNeighbors(int node1, int node2, int time, int num);
	void ComputeSmallCoNeighbors(int node1, int node2, int time, int num);
	int ComputeIntersectNum(int node1, int node2);
};


extern "C" {
	__declspec(dllexport) Graph* ConstructGraph();
	__declspec(dllexport) void InitialGraph(Graph* g, int n, int m);
	__declspec(dllexport) void Test(Graph* g, int& v1, int& v2);
	__declspec(dllexport) void GetRecordsNumBefore(Graph* g, int src, int time, int edge_id);
	__declspec(dllexport) void InitNeighbors(Graph* g, int node, int degree, int* neighbors, int* timestamps, int* edgeids);
	__declspec(dllexport) int GetInternalValue(Graph* g, int idx);
	__declspec(dllexport) void GetNeighborList(Graph* g, int start, int end, int num, int* tar_list);
	__declspec(dllexport) void GetTSList(Graph* g, int start, int end, int num, int* ts_list);
	__declspec(dllexport) void GetEdgeidList(Graph* g, int start, int end, int num, int* edgeid_list);
	__declspec(dllexport) void ComputeUnionAndIntersect(Graph* g, int node1, int node2, int time);
	__declspec(dllexport) void ComputeUnionAndIntersectWithEid(Graph* g, int node1, int node2, int eid);
	__declspec(dllexport) void ComputeSmallUnionAndIntersect(Graph* g, int node1, int node2, int time, int per_num);
	__declspec(dllexport) int GetUnionSize(Graph* g);
	__declspec(dllexport) int GetIntersectionSize(Graph* g);
	__declspec(dllexport) void PrintNeighbors(Graph* g, int node);
	__declspec(dllexport) void PrintNeighborWithTime(Graph* g, int node, int time);
	__declspec(dllexport) void GetNeighborWithTime(Graph* g, int node, int time, int num);
	__declspec(dllexport) int ComputeCoNeighbors(Graph* g, int node1, int node2, int time);
	__declspec(dllexport) void GetCoNeighbors(Graph* g, int* co_neighbors);
	__declspec(dllexport) void GetSortedCoNeighbors(Graph* g, int* co_neighbors, int num);
	__declspec(dllexport) void ComputeSortedCoNeighbors(Graph* g, int node1, int node2, int time, int num);
	__declspec(dllexport) int ComputeSmallCoNeighbors(Graph* g, int node1, int node2, int time, int num);
	__declspec(dllexport) int ComputeMaxCoNeiNum(Graph* g, int sample_num);
	__declspec(dllexport) int GetEdgeTime(Graph* g, int edge_idx);
	__declspec(dllexport) int GetDegreeBeforeEdgeid(Graph* g, int src, int edge_id);
	__declspec(dllexport) int GetTimeIntervalBeforeEdgeid(Graph* g, int src, int edge_id, int num);
	__declspec(dllexport) void Debug(Graph* g);
}
