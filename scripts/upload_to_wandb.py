import wandb
import pandas as pd
import json
import os

BASE_PATH = "/mnt/c/Users/steliosorfa/Desktop/Courses/ΠΤΥΧΙΑΚΗ/Research/experiments/tdc_drugres_baseline/results/tdc_drugres_baseline/baseline1"
for run_folder in os.listdir(BASE_PATH):

    run_path = os.path.join(BASE_PATH, run_folder)

    if not os.path.isdir(run_path):
        continue

    print("Uploading:", run_folder)

    run = wandb.init(
        project="Research",
        name=run_folder
    )

    history_file = os.path.join(run_path, "history.csv")
    if os.path.exists(history_file):

        df = pd.read_csv(history_file)

        for _, row in df.iterrows():
            run.log(row.to_dict())

    metrics_file = os.path.join(run_path, "metrics.json")
    if os.path.exists(metrics_file):

        with open(metrics_file) as f:
            metrics = json.load(f)

        run.summary.update(metrics)

    for img in [
        "loss_curve.png",
        "rmse_curve.png",
        "pearson_curve.png",
        "pred_vs_true.png"
    ]:

        img_path = os.path.join(run_path, img)

        if os.path.exists(img_path):
            run.log({img: wandb.Image(img_path)})

    model_path = os.path.join(run_path, "best_model.pt")

    if os.path.exists(model_path):

        artifact = wandb.Artifact(f"{run_folder}_model", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)

    run.finish()