"""Diagnose why mujoco cannot open a headless EGL display on this node.

Run it where the job runs, e.g. `srun -p gpu_h100 --gres=gpu:1 python scripts/egl_probe.py`,
and compare against a partition where rendering works. Native failures are
contained by running each device probe in a subprocess.
"""

import glob
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

print("host:", os.uname().nodename)
for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "MUJOCO_GL",
            "MUJOCO_EGL_DEVICE_ID", "__EGL_VENDOR_LIBRARY_FILENAMES",
            "__EGL_VENDOR_LIBRARY_DIRS", "LD_LIBRARY_PATH"):
  print(f"  {key}={os.environ.get(key)}")

print("\nglvnd EGL vendor files:")
for d in ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d"):
  for f in sorted(glob.glob(d + "/*.json")):
    print(" ", f, "->", open(f).read().split('"library_path"')[-1].strip(" :,}\n"))
print("\nlibEGL / libnvidia-egl candidates:")
out = subprocess.run("ldconfig -p | grep -Ei 'libEGL|nvidia-egl|libOSMesa|libnvidia-glcore'",
                     shell=True, capture_output=True, text=True).stdout
print(out or "  (none in ldconfig)")
print("DRI render nodes:", glob.glob("/dev/dri/*") or "none")
print("nvidia nodes:   ", sorted(glob.glob("/dev/nvidia*")))

PROBE = r"""
import sys
from mujoco.egl import egl_ext as EGL
devs = EGL.eglQueryDevicesEXT()
print("devices:", len(devs))
for i, d in enumerate(devs):
  disp = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, d, None)
  err = EGL.eglGetError()
  ok = None
  if disp != EGL.EGL_NO_DISPLAY:
    try:
      ok = EGL.eglInitialize(disp, None, None) == EGL.EGL_TRUE
    except Exception as e:
      ok = repr(e)
  print(f"  device {i}: display={'yes' if disp != EGL.EGL_NO_DISPLAY else 'NO'} "
        f"err=0x{err:x} initialize={ok}")
"""
print("\nmujoco device enumeration:")
r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120)
print(r.stdout, r.stderr[-1500:], f"(exit {r.returncode})")

for label, env in (("pinned (as train.py)", {}), ("unpinned", {"MUJOCO_EGL_DEVICE_ID": None})):
  e = {**os.environ, **{k: v for k, v in env.items() if v is not None}}
  for k, v in env.items():
    if v is None:
      e.pop(k, None)
  code = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); "
          "mujoco.Renderer(m, 64, 64).render(); print('render ok')")
  r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                     timeout=120, env=e)
  tail = (r.stdout + r.stderr).strip().splitlines()
  print(f"\nRenderer, {label}: exit {r.returncode}: {tail[-1] if tail else ''}")
