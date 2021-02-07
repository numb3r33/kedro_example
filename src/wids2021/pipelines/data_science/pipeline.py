from kedro.pipeline import Pipeline, node

from wids2021.pipelines.data_science.nodes import (
        evaluate_model,
        split_data,
        train_model
        )

def create_pipeline(**kwargs):
    return Pipeline(
            [
                node(
                    func=split_data,
                    inputs=['preprocessed_train', 'parameters'],
                    outputs=['X_train', 'X_test', 'y_train', 'y_test']
                    ),
                node(func=train_model, inputs=['X_train', 'y_train', 'parameters'], outputs='classifier'),
                node(
                    func=evaluate_model,
                    inputs=['classifier', 'X_test', 'y_test'],
                    outputs=None
                    )
            ]

            )
