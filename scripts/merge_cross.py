"""Merge the good episodes of both recordings into one dataset.

    $HOME/lerobot_env/bin/python scripts/merge_cross.py

The second recording session went into `rebot_pick_place` rather than `cross_v1`, so the good
episodes ended up split across two datasets. `rebot_pick_place` also still holds 60 episodes
from before the wrist-camera and lighting fixes, which must not be trained on: their wrist views
show nothing (the camera looked past its own gripper) and the scene is lit differently. On the
`front` camera, mean/std is 71/51 for the old episodes against 112/61 for every good one, i.e. a
visual domain shift on top of the dead wrist views.

  cross_v1                     51 eps   post-fix
  rebot_pick_place  60..90     31 eps   post-fix, the new session
  rebot_pick_place   0..59     60 eps   pre-fix, excluded

Writes a new dataset; both sources are left untouched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

CROSS = Path.home() / "zero_data" / "cross_v1"
RP = Path.home() / "zero_data" / "rebot_pick_place"
RP_KEEP = list(range(60, 91))            # the post-fix session only
OUT = Path.home() / "zero_data" / "cross_v2"


def main() -> None:
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if OUT.exists():
        shutil.rmtree(OUT)

    a = LeRobotDataset("zero/cross", root=CROSS)
    b = LeRobotDataset("zero/rp", root=RP, episodes=RP_KEEP)
    print(f"cross_v1          : {a.meta.total_episodes} eps, {a.num_frames} frames")
    print(f"rebot_pick_place  : eps {RP_KEEP[0]}-{RP_KEEP[-1]}, {b.num_frames} frames")

    # The task string is the language conditioning, so a mismatch would split the dataset into two
    # differently-conditioned halves. Compare the tasks the selected episodes actually use, not each
    # dataset's task table: rebot_pick_place's table still lists the old v1 string ("pick up the can
    # and place it in the tray") because episodes 0-59 used it, even though every episode being merged
    # here uses the current one.
    def used(ds) -> set[str]:
        names = list(ds.meta.tasks.index)
        return {names[i] for i in set(int(x) for x in ds.hf_dataset["task_index"])}
    ua, ub = used(a), used(b)
    if ua != ub or len(ua) != 1:
        raise SystemExit(f"task strings differ, refusing to merge:\n  {sorted(ua)}\n  {sorted(ub)}")
    print(f"task (both)       : {next(iter(ua))!r}")

    # Strip depth first. merge_datasets round-trips the parquet through pandas, and pyarrow cannot
    # convert the 224x224 nested float32 depth arrays:
    #   ArrowTypeError: Did not pass numpy.dtype object ... observation.depth.front
    # Training never reads depth (see scripts/make_train_view.py), so dropping it here costs nothing.
    # The originals keep their depth.
    from lerobot.datasets.dataset_tools import remove_feature, split_dataset
    depth = [k for k in a.meta.features if k.startswith("observation.depth.")]
    print(f"dropping          : {depth}")

    tmp_a, tmp_b, tmp_s = (Path("/tmp") / f"zero_merge_{n}" for n in ("a", "b", "split"))
    for d in (tmp_a, tmp_b, tmp_s):
        if d.exists():
            shutil.rmtree(d)

    print("\n1/4 stripping depth from cross_v1 ...")
    a2 = remove_feature(a, depth, output_dir=tmp_a, repo_id="zero/cross")
    print(f"    {a2.meta.total_episodes} eps")

    print(f"2/4 taking eps {RP_KEEP[0]}-{RP_KEEP[-1]} out of rebot_pick_place ...")
    parts = split_dataset(LeRobotDataset("zero/rp", root=RP), {"new": RP_KEEP}, output_dir=tmp_s)
    b_sel = parts["new"]
    print(f"    {b_sel.meta.total_episodes} eps")

    print("3/4 stripping depth from those ...")
    b2 = remove_feature(b_sel, depth, output_dir=tmp_b, repo_id="zero/rp")
    print(f"    {b2.meta.total_episodes} eps")

    print("4/4 merging ...")
    out = merge_datasets([a2, b2], "zero/cross", output_dir=OUT)
    print(f"\n{OUT}: {out.meta.total_episodes} eps, {out.meta.total_frames} frames")
    for d in (tmp_a, tmp_b, tmp_s):
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
