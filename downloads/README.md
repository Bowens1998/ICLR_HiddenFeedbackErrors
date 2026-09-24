# Material access

The [complete reproduction materials (1.35 GB)](https://osf.io/csgra/files?view_only=280626d2fc5e44c9b72620d0a2ffd0ee) are available through an anonymous OSF view-only link. No account is required.

1. On the OSF files page, choose **Download As Zip**, then extract the download.
2. Run `python assemble_reproduction.py` in the extracted directory. Python 3 is sufficient; no additional packages are needed.
3. Extract the resulting `when_predictions_become_inputs_anonymous_reproduction_v9.zip` and follow its code instructions.

The script joins 14 parts, checks every part, and verifies the complete ZIP against its original SHA256 checksum. Allow approximately 2.7 GB for the parts and reconstructed ZIP, plus space to extract the archive. You may also download the 14 parts and the two helper files individually into the same directory.

The [assembly script](assemble_reproduction.py) and [part manifest](REPRODUCTION_PARTS.json) are included here. [SHA256SUMS.txt](SHA256SUMS.txt) records the unchanged v9 archive checksums; [ARCHIVE_MANIFEST.json](../ARCHIVE_MANIFEST.json) describes the nine experiment archives inside the complete package.

The [code-only ZIP (4.5 MB)](when_predictions_become_inputs_anonymous_code_v9.zip) can be used for the saved-result analyses without downloading or assembling the larger package. Its contents and checksum are unchanged.
