
#%%
# # ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import pandas as pd
import great_expectations as gx


# ==========================================
# CREATE METADATA DATAFRAME
# ==========================================

def create_metadata_dataframe(X, y):

    metadata = []

    for img, label in zip(X, y):

        metadata.append({

            "height": img.shape[0],

            "width": img.shape[1],

            "channels": img.shape[2],

            "label": int(label),

            "min_pixel": float(img.min()),

            "max_pixel": float(img.max())

        })

    return pd.DataFrame(metadata)


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def validate_dataset(X, y):

    print("\nCreating Metadata DataFrame...")

    df = create_metadata_dataframe(X, y)

    os.makedirs("artifacts", exist_ok=True)

    df.to_csv(
        "artifacts/image_metadata.csv",
        index=False
    )

    # ==========================================
    # GX CONTEXT
    # ==========================================

    context = gx.get_context()

    # ==========================================
    # CREATE VALIDATOR
    # ==========================================

    validator = context.sources.pandas_default.read_dataframe(df)

    print("\nRunning Validation...")
    print("=" * 60)

    # ==========================================
    # EXPECTATIONS
    # ==========================================

    validator.expect_column_values_to_equal(
        "height",
        224
    )

    validator.expect_column_values_to_equal(
        "width",
        224
    )

    validator.expect_column_values_to_equal(
        "channels",
        3
    )

    validator.expect_column_values_to_be_in_set(
        "label",
        [0, 1]
    )

    validator.expect_column_values_to_be_between(
        "min_pixel",
        min_value=0,
        max_value=1
    )

    validator.expect_column_values_to_be_between(
        "max_pixel",
        min_value=0,
        max_value=1
    )

    # ==========================================
    # RUN VALIDATION
    # ==========================================

    results = validator.validate()

    print("\nValidation Results")
    print("=" * 60)

    print(
        "Validation Success :",
        results.success
    )

    print(
        "Success Percentage :",
        results.statistics["success_percent"],
        "%"
    )

    print(
        "Successful Expectations :",
        results.statistics["successful_expectations"]
    )

    print(
        "Total Expectations :",
        results.statistics["evaluated_expectations"]
    )

    return results
# %%
