#%%
# ==================================================
# IMPORT LIBRARIES
# ==================================================

import os
import mlflow
import mlflow.keras
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import (
    Sequential,
    Model
)

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    EfficientNetB0
)

# ==================================================
# MLFLOW CONFIGURATION
# ==================================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

# ==================================================
# COMMON TRAINING FUNCTION
# ==================================================

def train_and_log_model(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=10,
    batch_size=16
):

    # Create Artifacts Folder

    os.makedirs(
        "../artifacts",
        exist_ok=True
    )

    # Create Experiment

    mlflow.set_experiment(
        "Brain_Tumor_Detection"
    )

    # Start Run

    with mlflow.start_run(
        run_name=model_name
    ):

        # Compile Model

        model.compile(

            optimizer="adam",

            loss="binary_crossentropy",

            metrics=["accuracy"]
        )

        # Train Model

        history = model.fit(

            X_train,

            y_train,

            validation_data=(X_test, y_test),

            epochs=epochs,

            batch_size=batch_size,

            verbose=1
        )

        # Predictions

        y_prob = model.predict(X_test)

        y_pred = (
            y_prob > 0.5
        ).astype(int)

        # Evaluation Metrics

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        # Log Parameters

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_param(
            "epochs",
            epochs
        )

        mlflow.log_param(
            "batch_size",
            batch_size
        )

        # Log Metrics

        mlflow.log_metric(
            "accuracy",
            float(accuracy)
        )

        mlflow.log_metric(
            "f1_score",
            float(f1)
        )

        mlflow.log_metric(
            "roc_auc",
            float(roc_auc)
        )

        # Log Model

        mlflow.keras.log_model(

            model=model,

            name=model_name
        )

        # Save Model

        model.save(

            f"../artifacts/{model_name}.keras"
        )

        print("\n" + "="*50)

        print("Model :", model_name)

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC AUC : {roc_auc:.4f}"
        )

        print("="*50)

# ==================================================
# CNN MODEL
# ==================================================

def create_cnn():

    model = Sequential([

        Conv2D(
            32,
            (3,3),
            activation="relu",
            input_shape=(224,224,3)
        ),

        MaxPooling2D(),

        Conv2D(
            64,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D(),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            1,
            activation="sigmoid"
        )

    ])

    return model

# ==================================================
# MOBILENETV2
# ==================================================

def create_mobilenet():

    base_model = MobileNetV2(

        weights="imagenet",

        include_top=False,

        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = GlobalAveragePooling2D()(

        base_model.output
    )

    output = Dense(

        1,

        activation="sigmoid"
    )(x)

    model = Model(

        inputs=base_model.input,

        outputs=output
    )

    return model

# ==================================================
# RESNET50
# ==================================================

def create_resnet():

    base_model = ResNet50(

        weights="imagenet",

        include_top=False,

        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = GlobalAveragePooling2D()(

        base_model.output
    )

    output = Dense(

        1,

        activation="sigmoid"
    )(x)

    model = Model(

        inputs=base_model.input,

        outputs=output
    )

    return model

# ==================================================
# EFFICIENTNETB0
# ==================================================

def create_efficientnet():

    base_model = EfficientNetB0(

        weights="imagenet",

        include_top=False,

        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = GlobalAveragePooling2D()(

        base_model.output
    )

    output = Dense(

        1,

        activation="sigmoid"
    )(x)

    model = Model(

        inputs=base_model.input,

        outputs=output
    )

    return model

# ==================================================
# SHOW ALL RUNS
# ==================================================

def show_mlflow_runs():

    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(
        "Brain_Tumor_Detection"
    )

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id]
    )

    print("\nTotal Runs :", len(runs))

    for run in runs:

        print("\nRun Name :",
              run.info.run_name)

        print(
            run.data.metrics
        )


# %%
