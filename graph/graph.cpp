#include"graph.h"
//#define D(x) x
#define D(x) 



void Graph::GetNeighborList(int start, int offset, int num, int* tar_list) {
	D(cout<<"start: "<<start<<", offset: "<<offset<<", num: "<<num<<", edge_size: "<<timestamps.size()<<endl;)
	if (offset - start >= num) {
		start = offset - num;
		copy(targets.data() + start, targets.data() + offset, tar_list);
	}
	else {
		memset(tar_list, 0, sizeof(int) * (num - offset + start));
		copy(targets.data() + start, targets.data() + offset, tar_list + num - offset + start);
	}
//	std::copy(targets.data() + start, targets.data() + offset, tar_list);
}

void Graph::GetTSList(int start, int offset, int num, int* ts_list) {
	if (offset - start >= num) {
		start = offset - num;
		copy(timestamps.data() + start, timestamps.data() + offset, ts_list);
	}
	else {
		memset(ts_list, 0, sizeof(int) * (num - offset + start));
		copy(timestamps.data() + start, timestamps.data() + offset, ts_list + num - offset + start);
	}
//	std::copy(timestamps.data() + start, timestamps.data() + offset, ts_list);
}

void Graph::GetEdgeidList(int start, int offset, int num, int* edgeid_list) {
	if (offset - start >= num) {
		start = offset - num;
		std::copy(edge_idxs.data() + start, edge_idxs.data() + offset, edgeid_list);
	}
	else {
		memset(edgeid_list, 0, sizeof(int) * (num - (offset - start)));
		copy(edge_idxs.data() + start, edge_idxs.data() + offset, edgeid_list + num - (offset - start));
	}
//	std::copy(edge_idxs.data() + start, edge_idxs.data() + offset, edgeid_list);
}

void GetNeighborWithTime(Graph* g, int node, int time, int num) {

}

void Graph::RecordsBeforeNum(int src, int time, int edge_id) {
	debug = false;
	int src_start = offset[src];
	int src_end = offset[src + 1];
	int edge_offset = 0;
	//cout << "edge id " << edge_id << endl;
	if (edge_id < 0) {
		pair<int, int> key(src, time+1);
		if (timestamp2offset.find(key) == timestamp2offset.end()) {
			edge_offset = BinarySearch(src_start, src_end, time);
		}
		else {
			//edge_offset = timestamp2offset[make_pair(src, time)];
			edge_offset = timestamp2offset[key];
		}
	}
	else {
		//int time = timestamp[cut_edge_idx];
		edge_offset = edgeid2offset[make_pair(src, edge_id)]+1;
	}
	values[0] = src_start;
	values[1] = edge_offset;
	values[2] = edge_offset - src_start;
}


Graph* ConstructGraph() {
	Graph* g = new Graph();
	g->debug = false;
	return g;
}

void ComputeSortedCoNeighbors(Graph* g, int node1, int node2, int time, int num) {
	g->ComputeTimeSortedCoNeighbors(node1, node2, time, num);
}

int ComputeSmallCoNeighbors(Graph* g, int node1, int node2, int time, int num) {
	g->ComputeSmallCoNeighbors(node1, node2, time, num);
	return g->intersect_size;
}

int ComputeCoNeighbors(Graph* g, int node1, int node2, int time) {
	g->ComputeCoNeighbors(node1, node2, time);
	return g->intersect_size;
}

void GetCoNeighbors(Graph* g, int* nodes) {
	int idx = 0;
	assert(g->intersect_size == g->co_neighbors.size());
	for (int v : g->co_neighbors) {
		nodes[idx] = v;
		idx += 1;
	}
}

void GetSortedCoNeighbors(Graph* g, int* nodes, int num) {
	int idx = 0;
	for (int v : g->co_neighbors) {
		nodes[idx] = v;
		idx += 1;
		if (idx == num) {
			break;
		}
	}
	while (idx < num) {
		nodes[idx] = 0;
		idx += 1;
	}
}

void ComputeUnionAndIntersectWithEid(Graph* g, int node1, int node2, int eid) {
	g->ComputeUnionAndIntersectWithEid(node1, node2, eid);
}

void ComputeUnionAndIntersect(Graph* g, int node1, int node2, int time) {
	g->ComputeUnionAndIntersect(node1, node2, time);
}

int GetUnionSize(Graph* g) {
	return g->union_size;
}

int GetIntersectionSize(Graph* g) {
	return g->intersect_size;
}

int Graph::BinarySearch(int start, int end, int value) {
	int s = start;
	int e = end;
	int m = (s + e) / 2;
	while (s + 1 < e) {
		m = (s + e) / 2;
		if (timestamps[m] < value) {
			s = m;
		}
		else if (timestamps[m] >= value) {
			e = m;
		}
	}
	if (timestamps[s] < value && timestamps[e] >= value) {
		return s + 1;
	}
	else if (timestamps[s] >= value) {
		return s;
	}
	else if (timestamps[e] < value) {
		return e;
	}
	else {
		cout << "start: " << start << ", end: " << end << ", value: " << value << endl;
		for (int i = start; i < end; i++) {
			cout << timestamps[i] << " ";
		}
		cout << endl;
		cout << "this should not happend" << endl;
		exit(0);
	}
}


void Graph::ComputeUnionAndIntersectWithEid(int node1, int node2, int eid) {
	int start = offset[node1];
	int end = offset[node1 + 1];
	union_size = 0;
	intersect_size = 0;
	int edge_offset = edgeid2offset[make_pair(node1, eid)];
	int time = timestamps[edge_offset]+1;
	list<int> marked_nodes;
	for (int i = start; i < end; i++) {
		if (edge_idxs[i] == eid || timestamps[i] >= time) {
			break;
		}
		int neighbor = targets[i];
		if (flag[neighbor] == 0) {
			flag[neighbor] = 1;
			union_size += 1;
			marked_nodes.push_back(neighbor);
		}
	}
	start = offset[node2];
	end = offset[node2 + 1];
	for (int i = start; i < end; i++) {
		if (timestamps[i] >= time || edge_idxs[i] == eid) {
			break;
		}
		int neighbor = targets[i];
		switch (flag[neighbor])
		{
		case 0: {
			marked_nodes.push_back(neighbor);
			flag[neighbor] = 2;
			union_size += 1;
			break;
		}
		case 1: {
			flag[neighbor] = 3;
			intersect_size += 1;
			break;
		}
		default:
			break;
		}
	}
	for (int node : marked_nodes) {
		flag[node] = 0;
	}
}

int Graph::ComputeIntersectNum(int node1, int node2) {
	int count = 0;
	auto ptr1 = neighbor_sets[node1].begin();
	auto ptr2 = neighbor_sets[node2].begin();
	while (ptr1 != neighbor_sets[node1].end() && ptr2 != neighbor_sets[node2].end()) {
		if (*ptr1 < *ptr2) {
			ptr1++;
		}
		else if (*ptr1 > *ptr2) {
			ptr2++;
		}
		else {
			ptr1++;
			ptr2++;
			count++;
		}
	}
	return count;
}

void Graph::ComputeSmallUnionAndIntersect(int node1, int node2, int time, int num) {
	int node1_begin = offset[node1];
	int node1_end = offset[node1 + 1];
	int node2_begin = offset[node2];
	int node2_end = offset[node2 + 1];
	int node1_cut, node2_cut;
	if (timestamp2offset.find(make_pair(node1, time)) != timestamp2offset.end()) {
		node1_cut = timestamp2offset[make_pair(node1, time)] + 1;
	}
	else {
		node1_cut = BinarySearch(node1_begin, node1_end, time+1);
	}
	if (node1_begin < node1_cut - num) {
		node1_begin = node1_cut - num;
	}
	if (timestamp2offset.find(make_pair(node2, time)) != timestamp2offset.end()) {
		node2_cut = timestamp2offset[make_pair(node2, time)] + 1;
	}
	else {
		node2_cut = BinarySearch(node2_begin, node2_end, time  +1);
	}
	if (node2_begin < node2_cut - num) {
		node2_begin = node2_cut - num;
	}
	//cout << "node1_begin: " << node1_begin << " node1_cut: " << node1_cut << endl;
	//cout << "node2_begin: " << node2_begin << " node2_cut: " << node2_cut << endl;
	intersect_size = 0;
	union_size = 0;
	list<int> marked_nodes;
	for (int i = node1_cut - 1; i >= node1_begin; i--) {
		int neighbor = targets[i];
		//cout << "neighbor: " << neighbor <<" "<<flag[neighbor]<< endl;
		if (flag[neighbor] == 0) {
			flag[neighbor] = 1;
			marked_nodes.push_back(neighbor);
			union_size += 1;
		}
	}
	//cout << "############" << endl << endl;
	for (int i = node2_cut - 1; i >= node2_begin; i--) {
		int neighbor = targets[i];
		//cout << "neighbor: " << neighbor <<" "<<flag[neighbor]<< endl;
		switch (flag[neighbor])
		{
		case 0: {
			flag[neighbor] = 2;
			union_size += 1;
			marked_nodes.push_back(neighbor);
			break;
		}
		case 1: {
			flag[neighbor] = 3;
			intersect_size += 1;
			break;
		}
		default:
			break;
		}
	}
	for (int node : marked_nodes) {
		flag[node] = 0;
	}
}

void Graph::ComputeSmallCoNeighbors(int node1, int node2, int time, int num) {
	int node1_begin = offset[node1];
	int node1_end = offset[node1 + 1];
	int node2_begin = offset[node2];
	int node2_end = offset[node2 + 1];
	int node1_cut, node2_cut;
	if (timestamp2offset.find(make_pair(node1, time)) != timestamp2offset.end()) {
		node1_cut = timestamp2offset[make_pair(node1, time)]+1;
	}
	else {
		node1_cut = BinarySearch(node1_begin, node1_end, time+1);
	}
	if (node1_begin < node1_cut - num) {
		node1_begin = node1_cut - num;
	}
	if (timestamp2offset.find(make_pair(node2, time)) != timestamp2offset.end()) {
		node2_cut = timestamp2offset[make_pair(node2, time)]+1;
	}
	else {
		node2_cut = BinarySearch(node2_begin, node2_end, time+1);
	}
	if (node2_begin < node2_cut - num) {
		node2_begin = node2_cut - num;
	}
//	cout << "node1: " << node1 << ", node2: " << node2 << endl;
//	cout << "node1_begin: " << node1_begin << ", node1_end: " << node1_end << ", node1_cut: " << node1_cut << endl;
//	cout << "node2_begin: " << node2_begin << ", node2_end: " << node2_end << ", node2_cut: " << node2_cut << endl;
	intersect_size = 0;
	co_neighbors.clear();
	list<int> marked_nodes;
	for (int i = node1_cut - 1; i >= node1_begin; i --) {
		int neighbor = targets[i];
		if (flag[neighbor] == 0) {
			flag[neighbor] = 1;
			marked_nodes.push_back(neighbor);
		}
	}
	for (int i = node2_cut; i >= node2_begin; i--) {
		int neighbor = targets[i];
		switch (flag[neighbor])
		{
		case 0: {
			flag[neighbor] = 2;
			marked_nodes.push_back(neighbor);
			break;
		}
		case 1: {
			flag[neighbor] = 3;
			co_neighbors.push_back(neighbor);
			intersect_size += 1;
			break;
		}
		default:
			break;
		}
	}
	for (int node : marked_nodes) {
		flag[node] = 0;
	}
//	cout << "intersect size: " << intersect_size << endl;
}

void Graph::ComputeTimeSortedCoNeighbors(int node1, int node2, int time, int num) {
	int node1_begin = offset[node1];
	int node1_end = offset[node1 + 1];
	int node2_begin = offset[node2];
	int node2_end = offset[node2 + 1];
	int node1_cut, node2_cut;
	if (timestamp2offset.find(make_pair(node1, time)) != timestamp2offset.end()) {
		node1_cut = timestamp2offset[make_pair(node1, time)];
	}
	else {
		node1_cut = BinarySearch(node1_begin, node1_end, time);
	}
	if (timestamp2offset.find(make_pair(node2, time)) != timestamp2offset.end()) {
		node2_cut = timestamp2offset[make_pair(node2, time)];
	}
	else {
		node2_cut = BinarySearch(node2_begin, node2_end, time);
	}
	int node1_idx = node1_cut - 1;
	int node2_idx = node2_cut - 1;
	list<int> marked_nodes;
	intersect_size = 0;
	co_neighbors.clear();
	while (node1_idx >= node1_begin && node2_idx >= node2_begin) {
		int t1 = timestamps[node1_idx];
		int t2 = timestamps[node2_idx];
		if (t1 > t2) {
			int nei = targets[node1_idx];
			switch (flag[nei])
			{
			case 0: {
				flag[nei] = 1;
				marked_nodes.push_back(nei);
				break;
			}
			case 1: {
				break;
			}
			case 2: {
				flag[nei] = 3;
				co_neighbors.push_back(nei);
				break;
			}
			case 3: {
				break;
			}
			default:
				break;
			}
			node1_idx -= 1;
		}
		else if (t1 < t2) {
			int nei = targets[node2_idx];
			switch (flag[nei])
			{
			case 0: {
				flag[nei] = 2;
				marked_nodes.push_back(nei);
				break;
			}
			case 1: {
				flag[nei] = 3;
				co_neighbors.push_back(nei);
				break;
			}
			case 2: {
				break;
			}
			case 3: {
				break;
			}
			default:
				break;
			}
			node2_idx -= 1;
		}
		else {
			int nei1 = targets[node1_idx];
			switch (flag[nei1])
			{
			case 0: {
				flag[nei1] = 1;
				marked_nodes.push_back(nei1);
				break;
			}
			case 2: {
				flag[nei1] = 3;
				co_neighbors.push_back(nei1);
				break;
			}
			default:
				break;
			}
			int nei2 = targets[node2_idx];
			switch (flag[nei2])
			{
			case 0: {
				flag[nei2] = 2;
				marked_nodes.push_back(nei2);
				break;
			}
			case 1: {
				flag[nei2] = 3;
				co_neighbors.push_back(nei2);
				break;
			}
			default:
				break;
			}
			node1_idx -= 1;
			node2_idx -= 1;
		}
		if (co_neighbors.size() >= num) {
			break;
		}
	}
	for (int node : marked_nodes) {
		flag[node] = 0;
	}
}

void Graph::ComputeCoNeighbors(int node1, int node2, int time) {
	int start = offset[node1];
	int end = offset[node1 + 1];
	intersect_size = 0;
	co_neighbors.clear();
	list<int> marked_nodes;
	//cout << "time: " << time << endl;
	//cout << "start " << start << ", end " << end << endl;
	for (int i = start; i < end; i++) {
		//cout << "i: " << i << ", time: " << timestamps[i] << endl;
		if (timestamps[i] >= time) {
			break;
		}
		int neighbor = targets[i];
		if (flag[neighbor] == 0) {
			flag[neighbor] = 1;
			marked_nodes.push_back(neighbor);
		}
	//	cout << "flag[" << neighbor << "] = " << int(flag[neighbor]) << endl;
	}
	start = offset[node2];
	end = offset[node2 + 1];
//	cout << "start: " << start << ", end: " << end << endl;
	for (int i = start; i < end; i++) {
		//cout << "i: " << i << ", time: " << timestamps[i] << endl;
		if (timestamps[i] >= time) {
			break;
		}
		int neighbor = targets[i];
		//cout << "flag[" << neighbor << "] = " << int(flag[neighbor]) << endl;
		switch (flag[neighbor]) {
		case 0: {
			flag[neighbor] = 2;
			marked_nodes.push_back(neighbor);
			break;
		}
		case 1: {
			flag[neighbor] = 3;
			intersect_size += 1;
			co_neighbors.push_back(neighbor);
			break;
		}
		}
	}
	for (int node : marked_nodes) {
		flag[node] = 0;
	}
//	cout << "co neighbor num is " << intersect_size << endl;
}
void Graph::ComputeUnionAndIntersect(int node1, int node2, int time) {
	int start = offset[node1];
	int end = offset[node1 + 1];
	union_size = 0;
	intersect_size = 0;
	list<int> marked_nodes;
	for (int i = start; i < end; i++) {
		if (timestamps[i] >= time) {
			break;
		}
		int neighbor = targets[i];
		if (flag[neighbor] == 0) {
			flag[neighbor] = 1;
			union_size += 1;
			marked_nodes.push_back(neighbor);
		}
	}
	start = offset[node2];
	end = offset[node2 + 1];
	for (int i = start; i < end; i++) {
		if (timestamps[i] >= time) {
			break;
		}
		int neighbor = targets[i];
		switch (flag[neighbor])
		{
		case 0:
		{
			marked_nodes.push_back(neighbor);
			flag[neighbor] = 2;
			union_size += 1;
			break;
		}
		case 1: {
			flag[neighbor] = 3;
			intersect_size += 1;
			break;
		}
		default:
			break;
		}
	}
	for (int node : marked_nodes) {
		flag[node] = 0;
	}
}

void PrintNeighborWithTime(Graph* g, int node, int time) {
	int start = g->offset[node];
	int end = g->offset[node+1];
	for (int i = start; i < end; i++) {
		if (g->timestamps[i] >= time) {
			break;
		}
		cout << "i: " << i << ", dst: " << g->targets[i] << ", time: " << g->timestamps[i] << ", eid: " << g->edge_idxs[i] << ", map: "
			<< g->edgeid2offset[make_pair(node, g->edge_idxs[i])] << endl;
	}
}

void PrintNeighbors(Graph* g, int node) {
	int start = g->offset[node];
	int end = g->offset[node + 1];
	for (int i = start; i < end; i++) {
		cout <<"i: "<<i<< ", dst: " << g->targets[i] << ", time: " << g->timestamps[i] << ", eid: "<<g->edge_idxs[i]<<", map: "
			<<g->edgeid2offset[make_pair(node, g->edge_idxs[i])]<<endl;
	}
}

int ComputeMaxCoNeiNum(Graph* g, int sample_num) {
	for (int i = 0; i < g->node_num; i++) {
		int start = g->offset[i];
		int end = g->offset[i + 1];
		if (start == end)continue;
		int nei = g->targets[g->offset[i]];
		if (i < nei) {
			g->source_num = i + 1;
		}
		else {
			continue;
		}
	}
	g->target_num = g->node_num - g->source_num;
	cout << g->source_num << " source nodes, " << g->target_num << " destination nodes" << endl;
	int src_node_pair[2];
	int count = 0;
	g->max_co_neighbor_num = 0;
	while(count < sample_num){
		int src_node = rand() % g->source_num;
		int start = g->offset[src_node];
		int end = g->offset[src_node + 1];
		if (end - start == 1)continue;
		src_node_pair[0] = g->targets[start + rand() % (end - start)];
		src_node_pair[1] = src_node_pair[0];
		while (src_node_pair[1] == src_node_pair[0]) {
			src_node_pair[1] = g->targets[start + rand() % (end - start)];
		}
		int co_num = g->ComputeIntersectNum(src_node_pair[0], src_node_pair[1]);
		if (co_num > g->max_co_neighbor_num) {
			g->max_co_neighbor_num = co_num;
		}
		count += 1;
	}
	cout << "max_co_num: " << g->max_co_neighbor_num << endl;
	return g->max_co_neighbor_num;
}

void InitialGraph(Graph* g, int n, int m) {
	g->node_num = n;
	g->edge_num = m;
	cout << n << " nodes, " << m << " edges." << endl;
	g->offset.resize(n + 1, 0);
	g->targets.resize(m, 0);
	g->edge_idxs.resize(m, 0);
	g->timestamps.resize(m, 0);
	g->neighbor_sets.resize(n);
	g->flag.resize(n, 0);
	g->source_num = 0;
	g->target_num = INT_MAX;
}

//void FindBeforeStat(Graph* g, int src, int time, int edge_id, int& start, int& end) {

//}

int GetInternalValue(Graph* g, int idx) {
	return g->values[idx];
}

void GetRecordsNumBefore(Graph* g, int src, int time, int edge_id) {
	g->RecordsBeforeNum(src, time, edge_id);
}

void Test(Graph* g, int& v1, int& v2) {
	v1 = 1;
	v2 = 2;
}
void ComputeSmallUnionAndIntersect(Graph* g, int node1, int node2, int time, int per_num) {
	g->ComputeSmallUnionAndIntersect(node1, node2, time, per_num);
}


int CoNeighborNum(Graph* g, int node1, int node2) {
	auto ptr1 = g->neighbor_sets[node1].begin();
	auto ptr2 = g->neighbor_sets[node2].begin();
	int count = 0;
	while (ptr1 != g->neighbor_sets[node1].end() && ptr2 != g->neighbor_sets[node2].end()) {
		if (*ptr1 < *ptr2) {
			ptr1++;
		}
		else if (*ptr1 > *ptr2) {
			ptr2++;
		}
		else {
			count += 1;
			ptr1++;
			ptr2++;
		}
	}
	return count;
}

void InitNeighbors(Graph* g, int node, int degree, int* neighbors, int* timestamps, int* edgeids) {
	g->offset[node + 1] = g->offset[node] + degree;
	int start = g->offset[node];
	//cout << "node: "<<node<<", start: " << start << endl;
	for (int i = 0; i < degree; i++) {
		g->targets[start + i] = neighbors[i];
		g->timestamps[start + i] = timestamps[i];
		g->edge_idxs[start + i] = edgeids[i];
	}
	//copy(neighbors, neighbors + degree, g->targets.begin() + start);
	//copy(timestamps, timestamps + degree, g->timestamps.begin() + start);
	//copy(edgeids, edgeids + degree, g->edge_idxs.begin() + start);
	for (int i = 0; i < degree; i++) {
		g->neighbor_sets[node].insert(neighbors[i]);
	}
	pair<int, int> key;
	key.first = node;
	pair<int, int> nodeedge2idx_key;
	nodeedge2idx_key.first = node;
	for (int i = 0; i < degree; i++) {
		int ts = timestamps[i];
		key.second = ts;
	//	cout << "ts: " << ts << endl;
		if (g->timestamp2offset.find(key) == g->timestamp2offset.end()) {
			g->timestamp2offset[key] = start + i;
		}
		else {
			if (g->timestamp2offset[key] > start + i) {
				g->timestamp2offset[key] = start + i;
			}
		}
		nodeedge2idx_key.second = edgeids[i];
		g->edgeid2offset[nodeedge2idx_key] = start + i;
	}
//	for(auto ele : g->timestamp2offset){
//		if(ele.first.first == 10006 &&
//		ele.first.second == 397473){
//			
//			flt = true;
//			getchar();
//		}
//	}
	//if(flt == false){
//	cout << "####" << endl;
	//for (int i = 0; i < g->node_num; i++) {
	//	for (int j = g->offset[i]; j < g->offset[i + 1]; j++) {
	//		cout << "node " << i << " -> " << g->targets[j] << ", at time " << g->timestamps[j] << ", edge id: " << g->edge_idxs[j] << endl;
	//	}
	//}
	//for (auto k : g->timestamp2offset) {
	//	cout << "[" << k.first.first << ", " << k.first.second << "]: " << k.second<< endl;
	//}
}


void GetNeighborList(Graph* g, int start, int end, int num, int* tar_list) {
	g->GetNeighborList(start, end, num, tar_list);
}

void GetTSList(Graph* g, int start, int end, int num, int* ts_list) {
	g->GetTSList(start, end, num, ts_list);
}

void GetEdgeidList(Graph* g, int start, int end, int num, int* edgeid_list){
	g->GetEdgeidList(start, end, num, edgeid_list);
}


int GetEdgeTime(Graph* g, int edge_idx){
	return g->timestamps[edge_idx];
}

void Debug(Graph* g){
	g->debug = true;
}

int GetDegreeBeforeEdgeid(Graph* g, int src, int edge_id){
	int start = g->offset[src];
	int end = g->edgeid2offset[make_pair(src, edge_id)]+1;
	return end-start;
}

int GetTimeIntervalBeforeEdgeid(Graph* g, int src, int edge_id, int num){
	int start = g->offset[src];
	int end = g->edgeid2offset[make_pair(src, edge_id)]+1;
	if(end - start > num){
		start = end - num;
	}
	int interval = g->timestamps[end] - g->timestamps[start];
	return interval;
}
