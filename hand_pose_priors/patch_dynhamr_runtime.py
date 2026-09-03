#!/usr/bin/env python3
"""Apply the minimal no-HMP runtime patch to a pinned Dyn-HaMR checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_BLOCK = """from HMP.fitting import run_prior

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
"""

LAZY_CALL = """    if cfg.run_prior and not os.path.exists(os.path.join(out_dir, 'prior')):
        run_prior(cfg, dataset, out_dir, device, ['smooth_fit'], \\
        obs_data, hand_model, cfg, cfg.data, os.path.join(out_dir, 'prior'))
"""

PATCHED_LAZY_CALL = """    if cfg.run_prior and not os.path.exists(os.path.join(out_dir, 'prior')):
        from HMP.fitting import run_prior

        run_prior(cfg, dataset, out_dir, device, ['smooth_fit'], \\
        obs_data, hand_model, cfg, cfg.data, os.path.join(out_dir, 'prior'))
"""

RUN_MANO_TRANSLATION = """        transl=trans.reshape((B * seq_len, -1)),
"""

RUN_MANO_LOCAL_ONLY = """        # TouchAnything compatibility: mirror canonical left-hand geometry
        # before adding the shared world-coordinate translation.
        transl=None,
"""

RUN_MANO_MIRROR = """    joints[:, :, :, 0] = (2*is_right-1)*joints[:, :, :, 0]
    verts[:, :, :, 0] = (2*is_right-1)*verts[:, :, :, 0]

    return {
"""

RUN_MANO_MIRROR_THEN_TRANSLATE = """    joints[:, :, :, 0] = (2*is_right-1)*joints[:, :, :, 0]
    verts[:, :, :, 0] = (2*is_right-1)*verts[:, :, :, 0]

    # ``trans`` is expressed in the common world/camera coordinate system and
    # must not be reflected with the canonical left-hand mesh.
    joints = joints + trans[:, :T, None, :]
    verts = verts + trans[:, :T, None, :]

    return {
"""


def patch(root: Path) -> None:
    path = root / "dyn-hamr/run_opt.py"
    source = path.read_text(encoding="utf-8")
    changed = False
    if IMPORT_BLOCK in source:
        source = source.replace(IMPORT_BLOCK, "", 1)
        changed = True
    if LAZY_CALL in source:
        source = source.replace(LAZY_CALL, PATCHED_LAZY_CALL, 1)
        changed = True
    for eager_import in (
        "\nfrom vis.viewer import init_viewer\n",
        "\nfrom run_vis import run_vis\n",
    ):
        if eager_import in source:
            source = source.replace(eager_import, "\n", 1)
            changed = True
    viewer_anchor = "    if opts.vis_every > 0:\n        vis = init_viewer("
    if viewer_anchor in source:
        source = source.replace(
            viewer_anchor,
            "    if opts.vis_every > 0:\n"
            "        from vis.viewer import init_viewer\n\n"
            "        vis = init_viewer(",
            1,
        )
        changed = True
    run_vis_anchor = "    if cfg.run_vis:\n        run_vis("
    if run_vis_anchor in source:
        source = source.replace(
            run_vis_anchor,
            "    if cfg.run_vis:\n"
            "        from run_vis import run_vis\n\n"
            "        run_vis(",
            1,
        )
        changed = True
    if "from HMP.fitting import run_prior" not in source:
        raise RuntimeError("Dyn-HaMR HMP import patch was not applied")
    first_import = source.find("from HMP.fitting import run_prior")
    run_function = source.find("def run_opt(")
    if first_import < run_function:
        raise RuntimeError("Dyn-HaMR still imports HMP at module import time")
    for eager_import in (
        "from vis.viewer import init_viewer",
        "from run_vis import run_vis",
    ):
        import_position = source.find(eager_import)
        if 0 <= import_position < run_function:
            raise RuntimeError(f"Dyn-HaMR still imports {eager_import!r} eagerly")
    if changed:
        path.write_text(source, encoding="utf-8")
        print(f"[dynhamr-setup] Patched optional HMP imports in {path}")
    else:
        print(f"[dynhamr-setup] Runtime patch already present in {path}")

    scene_path = root / "dyn-hamr/optim/base_scene.py"
    scene_source = scene_path.read_text(encoding="utf-8")
    if "\nimport mano\n" in scene_source:
        scene_source = scene_source.replace("\nimport mano\n", "\n", 1)
        scene_path.write_text(scene_source, encoding="utf-8")
        print(f"[dynhamr-setup] Removed unused mano import in {scene_path}")
    elif "import mano" in scene_source:
        raise RuntimeError("Unexpected active mano import in Dyn-HaMR base_scene.py")

    optimizer_path = root / "dyn-hamr/optim/optimizers.py"
    optimizer_source = optimizer_path.read_text(encoding="utf-8")
    eager_output_import = "\nfrom vis.output import prep_result_vis, animate_scene\n"
    optimizer_changed = False
    if eager_output_import in optimizer_source:
        optimizer_source = optimizer_source.replace(eager_output_import, "\n", 1)
        optimizer_changed = True
    lazy_output_anchor = (
        "    def vis_result(self, res_dir, obs_data, vis=None, num_steps=-1):\n"
        "        if vis is None or self.vis_every < 0:\n"
        "            return\n"
    )
    if (
        lazy_output_anchor in optimizer_source
        and "        from vis.output import prep_result_vis, animate_scene\n"
        not in optimizer_source
    ):
        replacement = (
            lazy_output_anchor
            + "\n        from vis.output import prep_result_vis, animate_scene\n"
        )
        optimizer_source = optimizer_source.replace(lazy_output_anchor, replacement, 1)
        optimizer_changed = True
    if optimizer_changed:
        optimizer_path.write_text(optimizer_source, encoding="utf-8")
        print(f"[dynhamr-setup] Made visualization imports lazy in {optimizer_path}")
    top_level_section = optimizer_source.split("class StageOptimizer", 1)[0]
    if "from vis." in top_level_section:
        raise RuntimeError("Dyn-HaMR optimizer still imports visualization eagerly")

    losses_path = root / "dyn-hamr/optim/losses.py"
    losses_source = losses_path.read_text(encoding="utf-8")
    eager_bmc = "        self.bio_loss = BMCLoss(lambda_bl=1, lambda_rb=1, lambda_a=1)\n"
    guarded_bmc = (
        "        self.bio_loss = (\n"
        "            BMCLoss(lambda_bl=1, lambda_rb=1, lambda_a=1)\n"
        "            if self.loss_weights.get(\"bio\", 0.0) > 0.0\n"
        "            else None\n"
        "        )\n"
    )
    if eager_bmc in losses_source:
        losses_source = losses_source.replace(eager_bmc, guarded_bmc, 1)
        losses_path.write_text(losses_source, encoding="utf-8")
        print(f"[dynhamr-setup] Guarded zero-weight BMC setup in {losses_path}")
    elif guarded_bmc not in losses_source:
        raise RuntimeError("Unexpected Dyn-HaMR BMC loss setup")

    mano_utils_path = root / "dyn-hamr/body_model/utils.py"
    mano_utils_source = mano_utils_path.read_text(encoding="utf-8")
    mano_utils_changed = False
    if RUN_MANO_TRANSLATION in mano_utils_source:
        mano_utils_source = mano_utils_source.replace(
            RUN_MANO_TRANSLATION, RUN_MANO_LOCAL_ONLY, 1
        )
        mano_utils_changed = True
    if RUN_MANO_MIRROR in mano_utils_source:
        mano_utils_source = mano_utils_source.replace(
            RUN_MANO_MIRROR, RUN_MANO_MIRROR_THEN_TRANSLATE, 1
        )
        mano_utils_changed = True
    if RUN_MANO_LOCAL_ONLY not in mano_utils_source or RUN_MANO_MIRROR_THEN_TRANSLATE not in mano_utils_source:
        raise RuntimeError("Unexpected Dyn-HaMR left-hand translation implementation")
    if mano_utils_changed:
        mano_utils_path.write_text(mano_utils_source, encoding="utf-8")
        print(
            "[dynhamr-setup] Separated left-hand geometry mirroring from world "
            f"translation in {mano_utils_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    patch(Path(parser.parse_args().root).expanduser().resolve(strict=True))


if __name__ == "__main__":
    main()
