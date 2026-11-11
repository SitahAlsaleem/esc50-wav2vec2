# ESC-50 Sound Classification

Starter project for the ESC-50 and Wav2Vec 2.0 experiments. Put model implementations in `models/` and local dataset files in `data/`. Git tracks the directory placeholder and ignores the dataset, checkpoints, and results. The training and evaluation scripts currently check setup and seeding. Model training and metrics are pending.

## Colab

1. Upload `ESC50_Colab.ipynb` via **File → Upload notebook**.
2. Select **Runtime → Change runtime type → T4 GPU → Save**.
3. Run the GPU check. Enter the repository's HTTPS clone URL and run the setup cells. For a private repository, use a read-only GitHub token stored under `GITHUB_TOKEN` in Colab Secrets, and grant the notebook access to the secret. Give each collaborator their own access.

## GitHub

Create an empty private repository and run these commands from the extracted project directory in an authenticated terminal. Replace `OWNER` and `REPO` with your repository details.

```bash
git init -b main
git add .
git commit -m "Set up ESC-50 Colab project"
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

Invite your teammate through the repository's access settings. Each teammate can create a branch for their changes and open a pull request.

## Setup check

```bash
python -m pip install -r requirements.txt
bash run.sh 42
```

The default seed is 42. The launcher sets Python's hash seed before startup. The seed helper sets the Python, NumPy, and PyTorch random generators and configures cuDNN to prefer deterministic behavior. Results can still vary across hardware, library versions, or operations without a deterministic implementation.
