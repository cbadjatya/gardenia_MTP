// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "bfs.h"
#define THRUST_IGNORE_CUB_VERSION_CHECK

int main(int argc, char *argv[]) {
	std::cout << "Breadth-first Search by Xuhao Chen\n";
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "type <graph-file-prefix> "
              << "[source_id(0)]\n";
    std::cout << "Example: " << argv[0] << " web-Google.mtx 0\n";
    exit(1);
  }
  
  bool symmetrize = true;
  bool need_reverse = false;
  
  int source = 0;

  Graph g(argv[2], argv[1], symmetrize, need_reverse, &source); // return source with max degree

  

  int magic_val = -1;

  if (argc == 4) magic_val = atoi(argv[3]);
  auto m = g.V();

  std::vector<DistT> distances(m, MYINFINITY);
  // if(magic_val != -1)
  //   std::cout<<"--------------BFS OPT results for source "<<source<<" and magic val "<< magic_val <<"-----------------"<<endl;
  // BFSSolver(g, source, &distances[0], magic_val);

  BFSSolver(g, source, &distances[0], magic_val);
  // BFSVerifier(g, source, &distances[0]);


  return 0;
}

/*

get 5 randomly selected sources from 0 to m-1
get magic_val from input

*/
