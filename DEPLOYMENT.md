# Deployment

The serving image is self-sufficient. The checkpoint and the sample images are
baked in, `$PORT` is read at startup, and no volumes, compose file or `.env` are
required. Anything that can build a Dockerfile can host it.

```bash
docker run -p 8000:8000 logistics-exception-engine:1.0.0
```

## Measured footprint

Taken from the running container, not estimated.

| State | Memory |
| :--- | ---: |
| Idle, checkpoint loaded | 311 MiB |
| Peak during inference | 530 MiB |
| Image on disk | 2.79 GB |

CPU inference costs 1.2 to 1.7 s per image. The image is CPU-only on purpose: a
CUDA base would add several GB for no benefit to a reviewer on a laptop.

## Choosing a host

The 530 MiB peak is the number that decides this.

| Host | Free tier | Verdict |
| :--- | :--- | :--- |
| **Hugging Face Spaces** | 2 vCPU, 16 GB | **Recommended.** Fits comfortably, and it builds the Dockerfile rather than needing a 2.79 GB image push |
| Render | 512 MB | Too tight. Peak inference exceeds the limit, so it would be killed mid-request |
| Fly.io | 256 MB default | Needs a 1 GB machine configured before it will hold |
| Any VM with 1 GB+ | — | Works. `docker run` and a reverse proxy is the whole setup |

## Automated deployment

[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) pushes to a
Hugging Face Space on every commit to `main`.

Set up once. The workflow creates the Space itself if it does not exist, so the
only manual step is adding the credentials.

1. In GitHub, under **Settings → Secrets and variables → Actions**:

   | Kind | Name | Value |
   | :--- | :--- | :--- |
   | secret | `HF_TOKEN` | a Hugging Face token with **write** scope |
   | variable | `HF_USERNAME` | your Hugging Face username |
   | variable | `HF_SPACE` | the Space name |

Note the two tabs on that page: `HF_TOKEN` goes under **Secrets**, while
`HF_USERNAME` and `HF_SPACE` go under **Variables**. Mixing them up is the
usual cause of a skipped run.

Until those exist the job exits cleanly with a message rather than failing, so
the build stays green for anyone without a Hugging Face account.

The workflow assembles a fresh commit in a temporary directory rather than
pushing this repository's history, because `weights/best.pt` is a 63 MB regular
git blob and the Hub requires anything over 10 MB to be an LFS pointer. Pushing
the history would mean rewriting all of it into LFS. The Space is a deployment
target rather than a source of truth, so one replacing commit is the right shape
and stops its history accumulating a 63 MB LFS object per push.

## Manual deployment

If you would rather not wire up the Action:

```bash
pip install -U huggingface_hub && huggingface-cli login
git lfs install

git checkout -b deploy/hf-space

{
  printf -- '---\ntitle: Logistics Exception Engine\nemoji: 📦\n'
  printf -- 'colorFrom: blue\ncolorTo: gray\nsdk: docker\napp_port: 8000\npinned: false\n---\n\n'
  cat README.md
} > README.new && mv README.new README.md

git lfs track "weights/*.pt"
git add .gitattributes README.md
git commit -m "chore: add Space front matter"

git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
git push hf deploy/hf-space:main --force
git checkout main
```

The first build takes roughly 4 to 8 minutes on free hardware, mostly the CPU
torch install. Watch the Space's **Logs** tab.

## Once it is live

The endpoints are then reachable at:

```
https://<your-username>-<space-name>.hf.space/
https://<your-username>-<space-name>.hf.space/health
https://<your-username>-<space-name>.hf.space/api/v1/detect
https://<your-username>-<space-name>.hf.space/api/v1/reason
```

Add that URL to the top of [README.md](README.md) as a **Live demo** line, so a
reviewer can try the system without building anything.

## What has been verified

The workflow's assembly and run were tested locally rather than assumed. The
assembled tree was built with `docker build` and the resulting container served
`/health`, `/`, `/api/v1/samples`, `/api/v1/detect` and `/api/v1/reason`
correctly with an injected `$PORT`, no volumes and no environment file, peaking
at 466 MiB.

What has **not** happened is an actual push to a live Space, which needs a
Hugging Face account. Until that is done there is no public URL, and the
submission self-audit scores the deployment component on the manifest alone.
