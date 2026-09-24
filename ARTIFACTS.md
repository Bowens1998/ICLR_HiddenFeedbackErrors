# Reproduction materials

The [materials page](https://osf.io/csgra/files?view_only=280626d2fc5e44c9b72620d0a2ffd0ee) provides `hidden_feedback_code.zip` and `hidden_feedback_reproduction.zip`. The reproduction package includes the code and the nine experiment archives below. `SHA256SUMS.txt` identifies the two download packages; `ARCHIVE_MANIFEST.json` identifies the experiment archives.

## Experiment archives

| Filename | Size (MiB) | SHA256 |
|---|---:|---|
| `feedback_diagnostic_evidence.zip` | 19.1 | `b812f8ac12370d1034f8837d7314e3a72c227347843a507dad03257882fd5b21` |
| `feedback_diagnostic_end_to_end.zip` | 193.7 | `240da1314b7c3c173fe9bde912b6c4df0ac6e79ac5115a5722859491907a5c23` |
| `feedback_confirmation_evidence.zip` | 46.5 | `416727c7ce38a18a6cc264b69fe1b2661624834ad4eabc02f3a934e255efe593` |
| `velocity_readout_qualification.zip` | 13.9 | `1a9cdb0c6c7735c645684f7fcecddcfd3d83eb9b410df81f853a4e2cd58bed0f` |
| `feedback_reporting_supplement.zip` | 37.9 | `d47b16ec941bac724a3f29b4824ab7181aa82744462e1fb097de5b8e2faf44df` |
| `feedback_second_readout_scoring.zip` | 32.7 | `78cb2fda6beb2f9b1323d570d11ca5ff0fb702a4b473fcec9287ccb227ea5849` |
| `feedback_motion_mechanism.zip` | 442.5 | `af523333bc52320b3d488d44eec24086de140c5614f1998148c14ec0fd56de7d` |
| `independent_evaluation_readout.zip` | 205.5 | `891286d2689b22700e4f9db189b8ab9d4483d793c97ab958e7d237c442fdb423` |
| `pointmaze_confirmation.zip` | 287.9 | `0cf56d1f78fd947f494bd6ce39b8ccadbc39fe76cbee6b2ef1d43ac875e0927f` |

Place these archives in `archives/` and run `python verify_saved_evidence.py --list` to see the available analyses. `verify_prospective.py` uses the small arrays already included with the code. The raw-image demonstration is in `feedback_diagnostic_end_to_end.zip`, together with its checkpoints, readouts, normalizers, input images, and instructions.

The packages support saved-result analysis and the supplied raw-image demonstration. Complete historical training images, every training checkpoint, and external DINO-WM weights are not included. See `RUNTIMES.md` for recorded environments and execution times.
