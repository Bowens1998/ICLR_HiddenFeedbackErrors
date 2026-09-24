# Artifact inventory and access

## Current access status

The code repository includes scientific source, frozen protocols, editable figure sources, prospective saved arrays and acceptance records. The nine large evidence archives below are **not uploaded with this code commit**. No download URL is implied by their filenames. The complete reproduction ZIP is a separately prepared artifact.

The quick-start `verify_prospective.py` works with the supplied small arrays. Historical saved-evidence verification additionally needs the corresponding archive in `archives/`.

## Evidence archives

| Filename | Size (MiB) | SHA-256 |
|---|---:|---|
| `feedback_diagnostic_evidence.zip` | 19.1 | `b812f8ac12370d1034f8837d7314e3a72c227347843a507dad03257882fd5b21` |
| `feedback_diagnostic_end_to_end_v1.zip` | 193.7 | `240da1314b7c3c173fe9bde912b6c4df0ac6e79ac5115a5722859491907a5c23` |
| `feedback_confirmation_evidence_v1.zip` | 46.5 | `416727c7ce38a18a6cc264b69fe1b2661624834ad4eabc02f3a934e255efe593` |
| `velocity_readout_qualification_v1.zip` | 13.9 | `1a9cdb0c6c7735c645684f7fcecddcfd3d83eb9b410df81f853a4e2cd58bed0f` |
| `feedback_reporting_supplement_anonymous_v2.zip` | 37.9 | `d47b16ec941bac724a3f29b4824ab7181aa82744462e1fb097de5b8e2faf44df` |
| `feedback_second_readout_scoring_anonymous_v2.zip` | 32.7 | `78cb2fda6beb2f9b1323d570d11ca5ff0fb702a4b473fcec9287ccb227ea5849` |
| `feedback_motion_mechanism_anonymous_v3.zip` | 442.5 | `af523333bc52320b3d488d44eec24086de140c5614f1998148c14ec0fd56de7d` |
| `independent_evaluation_readout_anonymous_staged_v1.zip` | 205.5 | `891286d2689b22700e4f9db189b8ab9d4483d793c97ab958e7d237c442fdb423` |
| `pointmaze_confirmation_anonymous_v1.zip` | 287.9 | `0cf56d1f78fd947f494bd6ce39b8ccadbc39fe76cbee6b2ef1d43ac875e0927f` |

The machine-readable authoritative inventory is `ARCHIVE_MANIFEST.json`. Preserve these filenames and hashes. Upload the already anonymized supplied versions, rather than similarly named original packages.

## Recommended distribution

Keep source, protocols, small accepted arrays and figure generators in Git. Large frozen evidence archives can be versioned release attachments, with the same SHA-256 inventory. A complete reproduction ZIP is convenient for one-download use; individual archives avoid downloading unrelated evidence. These are alternative download layouts for the same evidence, not additional experiments.

A model/dataset hub is optional if larger raw datasets or independently reusable checkpoints are distributed later. Give any such release its own asset scope, acquisition instructions and version bindings. Do not treat an external hosting account as anonymous by default.

## Reproduction scope

These artifacts support reconstruction of saved experimental contrasts and a limited raw-input GPU demonstration. The demonstration includes three study-trained checkpoints, their readouts/normalizers and two fixed development examples. Complete historical training images, every study-trained checkpoint and external DINO-WM weights are not redistributed. Full raw-input regeneration of every historical experiment was deferred and is not claimed by this release. See `RUNTIMES.md` and the retained receipts for what was actually checked.

## Reviewer access

For a double-blind submission, use an anonymized mirror or anonymized supplementary attachment. The hosting owner, commit author, archive metadata and outgoing asset links must be considered together. The code payload does not supply or invent an anonymous reviewer link.
