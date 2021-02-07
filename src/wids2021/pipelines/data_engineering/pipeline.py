from kedro.pipeline import node, Pipeline
from wids2021.pipelines.data_engineering.nodes import (
        preprocess_train,
        preprocess_test,
        )

def create_pipeline(**kwargs):
    return Pipeline(
            [
                node(
                    func=preprocess_train,
                    inputs="train",
                    outputs="preprocessed_train",
                    name="preprocessing_train"
                ),
                node(
                    func=preprocess_test,
                    inputs="test",
                    outputs="preprocessed_test",
                    name="preprocessing_test"
                )
            ]
        )
