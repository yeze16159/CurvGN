## CurvGN

Curvature Graph Network

## Requirement:

* numpy
  
* torch
  
* torch_geometric
  
* networkx
  
* GraphRicciCurvature
 
 ## Run BenchMark:
 ```bash
 python pipelines.py
 ```
 
 ## Run synthetic experiment:
 ```bash
 cd syndata
 python Ranset_gen.py
 cd ..
 ./pipelines_heat.sh
 ```
 
 ## Cite:
 
 Please cite our paper in following if you use this code in your work:
 
 ```
@inproceedings{ze20curvature,
	title="Curvature Graph Network",
	author="Ze Ye and Kin Sum Liu and Tengfei Ma and Jie Gao and Chao Chen",
	booktitle="Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)",
	month="April",
	year="2020"
}
```
 
  
