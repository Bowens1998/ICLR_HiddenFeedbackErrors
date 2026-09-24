# Material access

The [code package](when_predictions_become_inputs_anonymous_code_v9.zip) contains the frozen v9 code, configurations, plotting inputs, and saved analysis arrays. The repository README documents the code structure and commands.

The complete reproduction package, `when_predictions_become_inputs_anonymous_reproduction_v9.zip` (1,346,412,781 bytes), contains the code and all nine experiment archives listed in [ARCHIVE_MANIFEST.json](../ARCHIVE_MANIFEST.json). An anonymous download endpoint for this larger file is being prepared; it is not included in the repository download.

After downloading that package, verify it using [SHA256SUMS.txt](SHA256SUMS.txt), then extract it into a separate directory. To use the archives with an existing code checkout, copy its `archives/` directory into the checkout.

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
```

The published v9 ZIP files are unchanged. Documentation updates in this repository do not alter their checksums.
