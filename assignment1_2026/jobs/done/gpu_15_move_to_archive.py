# repository clean-up (asked 17 Nov): moves what is no longer needed for the submission into
# archive/assignment1_2026_cleanup/ (same relative paths, same drive -> just renames, nothing deleted):
# model backups / candidates / four-negatives copies, the unused Task 5 sasrec_seed_1.pt, the old prediction backups
# (submission/output/before_*), the one-off notebooks and the caches; jobs/ and job_runner.ipynb are still in use by the
# runner -> moved by hand after the runner is stopped
import shutil
from pathlib import Path

assignment_dir = Path.cwd().parent                                  # .../assignment1_2026 (jobs run in submission/)
archive_dir = assignment_dir.parent / "archive" / "assignment1_2026_cleanup"

to_move = [path for pattern in ("trained_models_backup_*", "trained_models_*_candidates", "trained_models_task*_four_negatives")
           for path in sorted(assignment_dir.glob(pattern))]
to_move += [assignment_dir / "trained_models" / "task5" / "sasrec_seed_1.pt"]
to_move += sorted((assignment_dir / "submission" / "output").glob("before_*"))
to_move += [assignment_dir / "submission" / name for name in
            ("reblend_without_zero_ratings.ipynb", "retrain_bayesian_fms.ipynb", "retrain_tuned_models.ipynb")]
to_move += [assignment_dir / ".ipynb_checkpoints"] + sorted(assignment_dir.rglob("__pycache__"))

moved = 0
for path in to_move:
    if not path.exists():
        print(f"missing, skipped: {path.relative_to(assignment_dir)}", flush=True)
        continue
    target = archive_dir / path.relative_to(assignment_dir)
    if target.exists():
        print(f"already in the archive, skipped: {path.relative_to(assignment_dir)}", flush=True)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    moved += 1
    print(f"moved {path.relative_to(assignment_dir)}", flush=True)
print(f"RESULT moved {moved} of {len(to_move)} items to {archive_dir}", flush=True)
