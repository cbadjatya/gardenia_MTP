// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "sssp.h"

int main(int argc, char *argv[]) {
	std::cout << "Single Source Shortest Path by Xuhao Chen\n";
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <filetype> <graph-prefix> "
              << "[symmetrize(0/1)] [reverse(0/1)] [source_id(0)] [delta(1)]\n";
    std::cout << "Example: " << argv[0] << " mtx web-Google 0 1\n";
    exit(1);
  }
	int delta = 1;
  bool symmetrize = true;
  bool need_reverse = false;
  
  // std::cout<<symmetrize<<"\n";
  int source = 0;
  Graph g(argv[2], argv[1], symmetrize, need_reverse, &source);
 
  int magic_val = -1;

  if (argc == 4) magic_val = atoi(argv[3]);
  
  if(magic_val != -1)
    std::cout<<"\n\n--------------BFS OPT results for source "<<source<<" and magic val "<< magic_val <<"-----------------"<<endl;

  auto m = g.V();
  auto nnz = g.E();
  std::vector<DistT> distances(m, kDistInf);
	std::vector<DistT> wt(nnz, DistT(1));
  SSSPSolver(g, source, &wt[0], &distances[0], delta, magic_val);
  SSSPVerifier(g, source, &wt[0], &distances[0]); 
  return 0;
}

