# New Laptop Setup — Hindi ITN Electronic Grammar + Sparrowhawk Tests

Complete from-scratch guide to get this project running on a fresh Windows laptop,
including Docker (via Docker Desktop) and the Sparrowhawk ITN test pipeline.

> All the "WSL" commands run **inside the Ubuntu terminal**, not Windows PowerShell.
> Steps marked "(Windows)" run in **Windows PowerShell**.

---

## 1. Install WSL2 + Ubuntu (Windows PowerShell, as Administrator)

```powershell
wsl --install
wsl --set-default-version 2
```

Reboot when prompted. On first launch, Ubuntu asks you to create a username + password — set those (remember the password; it's your `sudo` password).

Verify:
```powershell
wsl --status        # should show Default Version: 2
```

---

## 2. Install Docker Desktop (Windows)

We use **Docker Desktop** (not the native WSL engine) because it handles WSL
networking/DNS/MTU automatically — avoids the TLS-timeout / DNS issues.

1. Download from https://www.docker.com/products/docker-desktop/ → **Windows AMD64**
2. Run the installer → keep **"Use WSL 2 instead of Hyper-V"** CHECKED
3. Launch Docker Desktop (whale icon in the system tray)
4. **Settings → General** → confirm **"Use the WSL 2 based engine"** is checked
5. **Settings → Resources → WSL Integration** → toggle **ON** for your **Ubuntu** distro
6. Click **Apply & Restart**

Verify (in a FRESH Ubuntu terminal):
```bash
docker context ls        # "desktop-linux" should be current (*)
docker run hello-world   # should pull + print "Hello from Docker!"
```

> Note: with Docker Desktop you do NOT run `sudo service docker start`.
> The daemon runs on the Windows side — just keep Docker Desktop open.

---

## 3. Get the code (WSL Ubuntu)

```bash
sudo apt update && sudo apt install -y git
git clone https://github.com/mayuris-00/NeMo-text-processing.git
cd NeMo-text-processing

# Your latest WIP work is on this branch:
git checkout hi-itn-electronic-backup-2026-06-04
```

Branches:
- `hi-itn-electronic-nvidia-base` — main working branch (NVIDIA-original base)
- `hi-itn-electronic-backup-2026-06-04` — WIP backup (electronic grammar work)

---

## 4. Set up Python + pynini (WSL)

`pynini` only builds on Linux — that's why grammar export happens in WSL.

```bash
sudo apt install -y python3 python3-pip
pip3 install pynini==2.1.5
pip3 install nemo_text_processing
```

(If you prefer an isolated env: `python3 -m venv ~/ntp-venv && source ~/ntp-venv/bin/activate`
before the `pip3 install` lines.)

---

## 5. Run the Sparrowhawk ITN test (WSL)

```bash
cd ~/NeMo-text-processing/tools/text_processing_deployment

bash export_grammars.sh \
  --GRAMMARS=itn_grammars \
  --LANGUAGE=hi \
  --INPUT_CASE=lower_cased \
  --MODE=test
```

What this does (chain):
1. `pynini_export.py` compiles the Hindi ITN grammars → `.far` files
2. `docker/build.sh` builds the `sparrowhawk` image (~30 min the FIRST time;
   it compiles protobuf/re2/sparrowhawk from source)
3. `docker/launch.sh` runs the container → executes
   `test_sparrowhawk_inverse_text_normalization.sh` → runs all the ITN tests,
   including `testITNElectronic` (your electronic test cases)

### Just the electronic test cases
The electronic cases live in:
`tests/nemo_text_processing/hi/data_inverse_text_normalization/test_cases_electronic.txt`

The `testITNElectronic` function in
`tests/nemo_text_processing/hi/test_sparrowhawk_inverse_text_normalization.sh`
runs them.

---

## Troubleshooting

- **`docker run hello-world` TLS timeout** → Docker Desktop not started, or WSL
  integration off. Open Docker Desktop, re-check Settings → Resources → WSL Integration.
- **Sparrowhawk build fails on a `git clone` / download** → network blip; just
  re-run the `export_grammars.sh` command (Docker caches completed layers).
- **`pynini` install fails** → make sure you're in WSL/Ubuntu, not Windows Python.
- **Rebuild the image from scratch** → add `FORCE_REBUILD=True` to the export command.
- **Re-use existing `.far` files (skip recompile)** → the script auto-detects them;
  to force overwrite, leave `OVERWRITE_CACHE=True` (default).

---

## Quick reference — daily workflow

```bash
# 1. Make sure Docker Desktop is running (Windows)
# 2. In WSL:
cd ~/NeMo-text-processing
git checkout hi-itn-electronic-backup-2026-06-04
cd tools/text_processing_deployment
bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE=hi --INPUT_CASE=lower_cased --MODE=test
```
