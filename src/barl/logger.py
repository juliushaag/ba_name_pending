
import numpy as np
import wandb


class Logger():

  def __init__(self, log_name : str, step_keys : set, run_name : str = None, tags : list[str] = None, config : dict = None):
    
    wandb.init(project=log_name, name=run_name, config=config, tags=tags)

    self._step_key_dict = dict()

    self._steps = { k : 0 for k in step_keys}

    for k in self._steps:
      wandb.define_metric(k)

  def register(self, key : str, step_key : str):
    wandb.define_metric(key, step_metric=step_key)

  def get_step(self, step_key : str):
    return self._steps[step_key]

  def _ensure_step_key(self, key: str, step_key: str):

    if step_key not in self._steps:
        raise KeyError(
            f"Unknown step key '{step_key}'. "
            f"Known keys: {list(self._steps.keys())}"
        )

    if self._step_key_dict.get(key) == step_key:
      return

    wandb.define_metric(key, step_metric=step_key)
    self._step_key_dict[key] = step_key

  def increment(self, steps):
    for k, v in steps.items():
      if k not in self._steps:
        raise KeyError(f"Unknown step key '{k}'")

      self._steps[k] += v
    
    wandb.log({
      **self._steps
    })

  def log(self, stats : dict):
    wandb.log({
      **stats
    })

  
  def log_video(self, key : str, video : list[np.ndarray]):
                        
    video = np.stack(video)

    if video.ndim != 4:
      raise ValueError(
          f"Expected video shape (T,H,W,C) or (T,C,H,W), got {video.shape}"
      )
    
    if video.shape[-1] == 3:
      video = video.transpose(0, 3, 1, 2)
      
    wandb.log({
        key : wandb.Video(
            video,
            fps=30,
            format="mp4"
        )
    })

  def finish(self):
    wandb.finish()