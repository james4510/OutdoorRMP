# GitHub upload commands

Run these inside the release folder:

```bash
cd github_release_radio_map_shadow_detection
git init
git add .
git commit -m "Add final evaluation scripts and representative results"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```
