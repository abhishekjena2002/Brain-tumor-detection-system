#%%
# ==========================================
# IMPORT LIBRARIES
# ==========================================

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ==========================================
# LOAD PRETRAINED MODEL
# ==========================================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224, 224, 3)
)

# Freeze pretrained layers

base_model.trainable = False

# ==========================================
# BUILD CUSTOM MODEL
# ==========================================

x = GlobalAveragePooling2D()(

    base_model.output
)

x = Dropout(0.3)(x)

output = Dense(

    1,

    activation="sigmoid"
)(x)

model = Model(

    inputs=base_model.input,

    outputs=output
)

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test, y_test),

    epochs=10,

    batch_size=16,

    verbose=1
)

# ==========================================
# TRAIN ACCURACY
# ==========================================

train_loss, train_accuracy = model.evaluate(

    X_train,

    y_train,

    verbose=0
)

print("\nTrain Accuracy :", round(train_accuracy, 4))

# ==========================================
# TEST ACCURACY
# ==========================================

test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0
)

print("Test Accuracy :", round(test_accuracy, 4))

# ==========================================
# PREDICTIONS
# ==========================================

y_prob = model.predict(X_test)

y_pred = (

    y_prob > 0.5

).astype(int)

# ==========================================
# F1 SCORE
# ==========================================

f1 = f1_score(

    y_test,

    y_pred
)

print("F1 Score :", round(f1, 4))

# ==========================================
# ROC AUC
# ==========================================

roc_auc = roc_auc_score(

    y_test,

    y_prob
)

print("ROC AUC :", round(roc_auc, 4))

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        y_pred
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save(

    "../artifacts/MobileNetV2.keras"
)

print("\nModel Saved Successfully")
# %%
