# Dawid-skene

Python (and numpy) implementation of the Dawid and Skene (1979).

## the test for running experiments in the paper.

```
$ pytest
```

## How to use

```python
from dawid_skene_model import list2array
from dawid_skene_model import DawidSkeneModel

class_num = 4
dataset = [
        [[0, 0, 0], [0], [0], [0], [0]],
        [[2, 2, 2], [3], [2], [2], [3]],
        [[0, 0, 1], [1], [0], [1], [1]]
]
dataset_tensor = list2array(class_num, dataset_list)
model = DawidSkeneModel(class_num, max_iter=45, tolerance=10e-100)
marginal_predict, error_rates, worker_reliability, predict_label = model.run(dataset_tensor)
```

### References:

- Dawid and Skene (1979). [Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm](https://www.jstor.org/stable/2346806?seq=1#metadata_info_tab_contents). Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28, No. 1, pp. 20-28.
- The implementation is based [dallascard/dawid_skene](https://github.com/dallascard/dawid_skene)