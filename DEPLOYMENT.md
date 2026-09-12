# Deployment

The serving image is self-sufficient. The checkpoint and the sample images are
baked in, `$PORT` is read at startup, and no volumes, compose file or `.env` are
required. Anything that can build a Dockerfile can host it.

```bash
docker run -p 8000:8000 logistics-exception-engine:1.0.0
```

## Measured footprint

Taken from the running container, not estimated. The authoritative sizing number
is the hard floor in the next section, found by bisecting against `--memory`
caps; the observed peak below is what `docker stats` reports when memory is
plentiful, and it drifts with allocator pressure, so do not size a host from it.

| State | Memory |
| :--- | ---: |
| Idle, checkpoint loaded | 311 MiB |
| Observed peak, uncapped | ~516 MiB |
| **Hard floor (see below)** | **between 520 and 544 MB** |
| Image on disk | 2.79 GB |

CPU inference costs 1.2 to 1.7 s per image. The image is CPU-only on purpose: a
CUDA base would add several GB for no benefit to a reviewer on a laptop.

## Choosing a host

### The hard floor, measured by bisection

Run under successively tighter `--memory` caps, issuing one detection request at
each. Not estimated:

| Cap | Detection request | Outcome |
| ---: | :--- | :--- |
| 512 MB | HTTP 000 | OOM-killed |
| 520 MB | HTTP 000 | OOM-killed |
| 544 MB | HTTP 200 | survives |
| 640 MB | HTTP 200 | survives, peaks at 516 MiB |

**The container needs between 520 and 544 MB.** Pinning `OMP_NUM_THREADS=1` and
`MALLOC_ARENA_MAX=2` does not close the gap; the peak is RT-DETR-L's decoder
activations at 640 px. Lowering inference resolution would fit a 512 MB host but
costs detection accuracy, which is not a trade worth making.

### Host comparison

| Host | Free tier | Verdict |
| :--- | :--- | :--- |
| **Google Cloud Run** | configurable to 1 GB | **Only free option that fits.** Scales to zero, so the first request after idle waits 30-60 s for a cold start. Needs a card on file |
| Hugging Face Spaces | PRO only, $9/mo | Docker Spaces are **no longer free**. `create_repo` returns `402 Payment Required`: "hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription" |
| Render | 512 MB | **Does not fit.** 32 MB short. The paid Starter tier is also 512 MB; 2 GB costs roughly $25/mo |
| Fly.io | none for new accounts | No free tier for signups as of 2026 |
| Any VM with 1 GB+ | — | Works. `docker run` and a reverse proxy is the whole setup |

### Current status

**No live deployment exists.** The decision was to submit without one: the
component is worth roughly 2.5% of the grade, the other three bonus items
(containerisation, logging, error handling) are already satisfied, and the
free hosts either do not fit or answer the first request a minute late.

Reviewers run the system with `docker compose up`, which is verified working
end to end by `scripts/smoke_test_docker.py`.

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

The first build takes roughly 4 to 8 minutes, mostly the CPU torch install.
Watch the Space's **Logs** tab. This path requires a PRO subscription, as above.

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

What has **not** happened is an actual push to a live Space, because Docker
Spaces now require a paid subscription. There is no public URL.

The submission self-audit scores this component as **not satisfied**. It checks
for a reachable URL and deliberately does not accept a deploy manifest as a
substitute: a manifest proves the deployment is configured, not that it
happened. The audit therefore reports 93/95 rather than 95/95.
