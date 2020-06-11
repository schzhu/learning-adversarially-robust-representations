## Representation Robustness Evaluations
Our implementation is based on code from MadryLab's [robustness package](https://github.com/MadryLab/robustness) and Devon Hjelm's [Deep InfoMax](https://github.com/rdevon/DIM). For all the scripts, we assume the working directory to be the root folder of our code.
#### Get ready a pre-trained model
We have two methods to pre-train a model for evaluation. 
**Method 1**: Follow instructions from MadryLab's [robustness package](https://github.com/MadryLab/robustness) to train a **standard model** or a **robust model** with a given PGD setting.
For example, to train a robust ResNet18 with l-inf constraint of eps 8/255
```bash
python -m robustness.main --dataset cifar \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch resnet18 \
--epoch 150 \
--adv-train 1 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--exp-name resnet18_adv
```
**Method 2**: Use our wrapped code and set **task=train-model**.
Optional commands:
* -\-classifier-loss = **robust** (adversarial training) / **standard** (standard training)
* -\-arch = **baseline_mlp** (baseline-h with last two layer as mlp) /  **baseline_linear** (baseline-h with last two layer as linear classifier) / **vgg16** / ...

Our results presented in Figure 1 and 2 use model architecture: baseline_mlp, resnet18, vgg16, resnet50, DenseNet121.
For example, to train a baseline-h model with l-inf constraint of eps 8/255
```bash
python main.py --dataset cifar \
--task train-model \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch baseline_mlp \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--classifier-loss robust \
--exp-name baseline_mlp_adv
```

To parse the store file, run
```python
from cox import store
s = store.Store('/path/to/model/parent-folder', 'model-folder')
print(s['logs'].df)
s.close()
```
&nbsp;

#### Evaluate the representation robustness (Figure 1, 2, 3)
Set **task=estimate-mi** to load a pre-trained model and test the mutual information between input and representation. By subtracting the normal-case and worst-case mutual information we have the representation vulnerability.
Optional commands:
* -\-estimator-loss = **worst** (worst-case mutual information estimation) / **normal** (normal-case mutual information estimation)

For example, to test the worst-case mutual information of ResNet18, run
```bash
python main.py --dataset cifar \
--data /path/to/dataset \
--out-dir /path/to/output \
--task estimate-mi \
--representation-type layer \
--estimator-loss worst \
--arch resnet18 \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--resume /path/to/saved/model/checkpoint.pt.best \
--exp-name estimator_worst__resnet18_adv \
--no-store
```
or to test on the baseline-h, run
```bash
python main.py --dataset cifar \
--data /path/to/dataset \
--out-dir /path/to/output \
--task estimate-mi \
--representation-type layer \
--estimator-loss worst \
--arch baseline_mlp \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--resume /path/to/saved/model/checkpoint.pt.best \
--exp-name estimator_worst__baseline_mlp_adv \
--no-store
```
&nbsp;

#### Learn Representations
Set **task=train-encoder** to learn a representation using our training principle. For train by worst-case mutual information maximization, we can use other lower-bound of mutual information as surrogate for our target, which may have slightly better empirical performance (e.g. nce). Please refer to arxiv.org/abs/1808.06670 for more information.
Optional commands:
* -\-estimator-loss = **worst** (worst-case mutual information maximization) / **normal** (normal-case mutual information maximization)
* -\-va-mode = **dv** (Donsker-Varadhan representation) / **nce** (Noise-Contrastive Estimation) / **fd** (fenchel dual representation)
* -\-arch = **basic_encoder** ([Hjelm et al.](https://arxiv.org/abs/1808.06670)) / ...


Example:
```bash
python main.py --dataset cifar \
--task train-encoder \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch basic_encoder \
--representation-type layer \
--estimator-loss worst \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--exp-name learned_encoder
```
&nbsp;

#### Test on Downstream Classifications (Figure 4, 5, 6; Table 1, 3)
Set **task=train-classifier** to test the classification accuracy of learned representations. 
Optional commands:
* -\-classifier-loss = **robust** (adversarial classification) / **standard** (standard classification)
* -\-classifier-arch = **mlp** (mlp as downstream classifier) /  **linear** (linear classifier as downstream classifier)

Example:
```bash
python main.py --dataset cifar \
--task train-classifier \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch basic_encoder \
--classifier-arch mlp \
--representation-type layer \
--classifier-loss robust \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--resume /path/to/saved/model/checkpoint.pt.latest \
--exp-name test_learned_encoder
```

