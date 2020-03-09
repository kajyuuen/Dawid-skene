import pytest
import numpy as np

from dawid_skene_model import list2array
from dawid_skene_model import DawidSkeneModel

class TestDawidSkeneModel:
    def setup(self):
        self.class_num = 4
        self.dataset_list = [
            [[0, 0, 0], [0], [0], [0], [0]],
            [[2, 2, 2], [3], [2], [2], [3]],
            [[0, 0, 1], [1], [0], [1], [1]],
            [[1, 1, 1], [2], [0], [1], [0]],
            [[1, 1, 1], [2], [1], [1], [1]],
            [[1, 1, 1], [2], [2], [1], [1]],
            [[0, 1, 1], [1], [0], [0], [0]],
            [[2, 2, 2], [2], [3], [2], [2]],
            [[1, 1, 1], [1], [1], [1], [2]],
            [[1, 2, 1], [1], [1], [1], [2]],
            [[3, 3, 3], [3], [3], [3], [3]],
            [[1, 1, 1], [2], [2], [3], [2]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[1, 1, 1], [2], [1], [0], [1]],
            [[0, 1, 0], [0], [0], [0], [0]],
            [[0, 0, 0], [1], [0], [0], [0]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[1, 1, 1], [1], [1], [1], [0]],
            [[1, 1, 1], [0], [2], [1], [1]],
            [[1, 1, 1], [1], [1], [1], [1]],
            [[1, 1, 1], [1], [1], [1], [0]],
            [[1, 1, 1], [2], [1], [1], [1]],
            [[1, 1, 0], [1], [1], [1], [1]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[1, 2, 1], [1], [1], [1], [1]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[0, 0, 1], [0], [0], [1], [0]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[2, 2, 2], [2], [1], [2], [2]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[1, 1, 1], [1], [1], [1], [1]],
            [[1, 1, 1], [2], [1], [2], [1]],
            [[3, 2, 2], [3], [2], [3], [2]],
            [[1, 1, 0], [1], [1], [2], [1]],
            [[1, 2, 1], [2], [1], [2], [2]],
            [[2, 2, 2], [2], [3], [2], [1]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[0, 0, 0], [0], [0], [0], [0]],
            [[0, 1, 0], [1], [0], [0], [0]],
            [[1, 2, 1], [1], [1], [1], [1]],
            [[0, 1, 0], [0], [0], [0], [0]],
            [[1, 1, 1], [1], [1], [1], [1]]
    ]

    def test_dawid_skene_model(self):
        dataset_tensor = list2array(self.class_num, self.dataset_list)
        model = DawidSkeneModel(self.class_num, max_iter=45, tolerance=10e-100)
        marginal_predict, error_rates, worker_reliability, predict_label = model.run(dataset_tensor)
        
        # Table 2：Marginal probabilities
        for i, merginal_prob in enumerate([0.40, 0.42, 0.11, 0.07]):
            assert round(marginal_predict[i], 2) == pytest.approx(merginal_prob)
        
        # Table 3: Observer 1
        observer_1 = [
            [0.89, 0.11, 0.00, 0.00],
            [0.07, 0.88, 0.05, 0.00],
            [0.00, 0.34, 0.66, 0.00],
            [0.00, 0.00, 0.56, 0.44]
        ]
        for true_response, paper_true_response in zip(error_rates[0], observer_1):
            assert [round(x, 2) for x in true_response] == paper_true_response
            
        # Table 4: Final estimates of indicator variables for each patient
        paper_predict_label = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        for i in range(6):
            assert [round(x, 2) for x in predict_label[i]] == paper_predict_label[i]
