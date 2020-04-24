#nohup python -u plot_graph.py ConvZePlain 1000 > Rand_Plain.log &
#nohup python -u plot_graph.py ConvZeNMLP 1000 > Rand_NMLP.log &
#nohup python -u plot_graph.py ConvZeRand 1000 > Rand_Rand.log &
nohup python -u plot_graph.py ConvZeRandn 1000 > Rand_Rand.log &
#nohup python -u plot_graph.py Convgat 1000 > Rand_gat.log &
#nohup python -u plot_graph.py Convgcn 1000 > Rand_gcn.log &
#nohup python -u plot_graph.py Convgcn_mlp 1000 > Rand_gcn_mlp.log &

