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
@inproceedings{Ze2020curv,
  title={Curvature Graph Network},
  author={Ze Ye and Kin Sum Liu and Tengfei Ma and Jie Gao and Chao Chen},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```
 
  
