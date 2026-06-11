import wandb
api = wandb.Api()
run = api.run("/bakirhabib-t-l-com-paris/pufferMO-DST/runs/bzk184jo")
history = run.history()
history.to_csv("metrics.csv", index=False)
