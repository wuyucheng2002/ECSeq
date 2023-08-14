## Efficient User Behavior Sequential Learning with Compressed Graph Neural Networks

### Instructions 

1. Download the datasets in `data` folder and unzip it.

2. Install dependencies:

   ```shell
   pip install -r requirements.txt
   ```


3. Run *ECSeq* on *Bike* dataset:

    ```shell
    pip install -r requirements.txt
    python ECSeq_traffic.py --dataset=bike_nyc --method=ECSeq
    ```

    The results will be printed, and also saved in `results.txt`; models will be saved in `model` folder; figures of evaluation results will be saved in `fig` folder.

    If you want to retrain the models, please empty the `model` folder first.

### Arguments Description

| Name         | Default value | Description                                                  |
| ------------ | ------------- | ------------------------------------------------------------ |
| dataset      | bike_nyc      | Dataset file name, can be chosen from {'bike_nyc', 'pems_bay'}. |
| seq_backbone | lstm          | Sequence embedding extractor backbone, can be chosen from {'lstm', 'transformer'}. |
| gnn_backbone | GraphSAGE     | Graph mining backbone, can be chosen from {'GraphSAGE', 'GraphSAGE_max', 'GCN', 'GAT'}. |
| method       | ECSeq         | 'ECSeq': use ECSeq framework; 'batchGNN': don't use ECSeq and train GNN on graph batchs. |
| compress     | kmeans_no     | Graph compression algorithm, can be chosen from {'kmeans', 'AGC', 'Grain', 'Loukas'}. |
| n_cluster    | 100           | Number of clusters/new nodes.                                |

### References

- pytorch_geometric: https://github.com/pyg-team/pytorch_geometric
- UCTB: https://github.com/uctb/UCTB
- R-transformer: https://github.com/DSE-MSU/R-transformer
- graph coarsening: https://github.com/loukasa/graph-coarsening
- AGC: https://github.com/karenlatong/AGC-master
- Grain: https://github.com/zwt233/Grain