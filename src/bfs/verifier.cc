// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include <iostream>
#include <vector>
#include <queue>
#include "bfs.h"
#include "timer.h"

void BFSVerifier(Graph &g, int source, DistT *depth_to_test) {
	std::cout << "Verifying...\n";
  	auto m = g.V();
	auto nnz = g.E();
	auto h_row_offsets = g.out_rowptr();
	auto h_column_indices = g.out_colidx();
	vector<DistT> depth(m, MYINFINITY);
	vector<int> to_visit;
	Timer t;
	t.Start();
	depth[source] = 0;
	to_visit.reserve(m);
	to_visit.push_back(source);
	queue<int> q;
	q.push(source);
	while(!q.empty()) {
		int src = q.front();
		q.pop();
		for (int off = h_row_offsets[src];off < h_row_offsets[src+1];off++) {
			int dst = h_column_indices[off];
			if (depth[dst] == MYINFINITY) {
				depth[dst] = depth[src] + 1;
				q.push(dst);

			}
		}
	}
	t.Stop();
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());

	// Report any mismatches
	bool all_ok = true;
	for (int n = 0; n < m; n ++) {
		// std::cout << n << ": " << depth_to_test[n] << " -- " << depth[n] << std::endl;
		if (depth_to_test[n] != depth[n]) {
			std::cout << n << ": " << depth_to_test[n] << " != " << depth[n] << std::endl;
			all_ok = false;
		}
	}
	if(all_ok) printf("Correct\n");
	else printf("Wrong\n");
}

