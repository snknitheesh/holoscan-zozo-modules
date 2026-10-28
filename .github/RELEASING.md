# Releasing a module

Each module is released on its own, as a GitHub Release with its Debian packages. Pushing a tag
`<name>-v<major>.<minor>.<patch>` on a commit of `main`, such as `apriltag-v1.0.0`, runs
[`workflows/release.yml`](workflows/release.yml):

| Job | Runs on | What it does |
| --- | --- | --- |
| `Package amd64` | Self-hosted x86_64 runner (`self-hosted, linux, x64, gpu`) | Builds the module's `.deb` in the CI image, then installs it with APT in a fresh CI container and builds and runs the module's `tests/installed_consumer/` against it |
| `Package arm64` | Self-hosted Jetson Thor (`self-hosted, linux, arm64, jetson, thor`) | The same, natively on arm64 |
| `Publish` | GitHub-hosted `ubuntu-24.04` | Creates the GitHub Release for the tag, or updates it, with both packages and `SHA256SUMS` |

The release has, for `holoscan-apriltag` 1.0.0:

```text
https://github.com/snknitheesh/holoscan-zozo-modules/releases/download/apriltag-v1.0.0/holoscan-apriltag_1.0.0_amd64.deb
https://github.com/snknitheesh/holoscan-zozo-modules/releases/download/apriltag-v1.0.0/holoscan-apriltag_1.0.0_arm64.deb
https://github.com/snknitheesh/holoscan-zozo-modules/releases/download/apriltag-v1.0.0/SHA256SUMS
```

## Cut a release

1. In a pull request, set the new version in the module's `metadata.json` (`module.version`), in
   its `CMakeLists.txt` (`project(... VERSION ...)`), and in `binary_packages.install_commands`,
   which name the release's package (see [modules/README.md](../modules/README.md#releases-and-binary_packages)).
   Merge it.
2. Tag the merged commit of `main`, and push the tag alone:

   ```bash
   git fetch origin main
   git tag -a apriltag-v1.0.0 origin/main -m "holoscan-apriltag 1.0.0"
   git push origin refs/tags/apriltag-v1.0.0
   ```

3. Follow the run under **Actions > Release**. When it succeeds, the release is under
   **Releases**. Check a package where it is meant to be used:

   ```bash
   release=https://github.com/snknitheesh/holoscan-zozo-modules/releases/download/apriltag-v1.0.0
   deb="holoscan-apriltag_1.0.0_$(dpkg --print-architecture).deb"
   curl -fL -O "$release/$deb" -O "$release/SHA256SUMS"
   sha256sum --check --ignore-missing SHA256SUMS
   sudo apt install "./$deb"
   ```

## What the workflow checks

[`scripts/release.py`](scripts/release.py) does the work in the CI image, with the module's own
packaging (`pkg/holoscan-<name>/`). Before it builds anything, the tag must:

- have the form `<name>-v<major>.<minor>.<patch>`, without a pre-release or build suffix;
- name a folder `modules/<name>/` whose `metadata.json` has `module.name` `holoscan-<name>` and
  `module.version` equal to the tag's version;
- be on `main`, so that only reviewed code is released.

It then configures the module with `-DPKG_holoscan_<name>=ON` (`-` becomes `_`), which builds for
every major GPU architecture, runs CPack with the configuration that writes, and checks that the
package is `holoscan-<name>_<version>_<arch>.deb` with that package name, version (the CMake
project's) and architecture. In a second, fresh container, as root, it installs the package with
`apt-get install` (which also installs its dependencies, `holoscan (>= 5.0.0)` among them, from
the image's NVIDIA repositories), and builds and runs `tests/installed_consumer/` against `/usr`
alone. Any failure fails the job, and nothing is published.

To run the same steps on a machine with the CI image (`holoscan-modules-ci:local`) and a GPU:

```bash
docker run --rm --init --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:$PWD" -w "$PWD" holoscan-modules-ci:local \
  python3 -B .github/scripts/release.py package apriltag --version 1.0.0 --out build-release/dist
docker run --rm --init --gpus all -v "$PWD:$PWD:ro" -w "$PWD" holoscan-modules-ci:local \
  python3 -B .github/scripts/release.py consume apriltag \
  --deb build-release/dist/holoscan-apriltag_1.0.0_amd64.deb
```

## If a release fails

- **The run failed** (a test, the network): fix the cause on `main` if it is in the code, then
  delete the tag and push it again on the fixed commit:

  ```bash
  git push origin :refs/tags/apriltag-v1.0.0
  git tag -d apriltag-v1.0.0
  ```

  A transient failure needs only **Re-run failed jobs**.
- **The release exists and must be rebuilt**: re-running the workflow, or pushing the tag again,
  replaces the packages and `SHA256SUMS` of the existing release (`gh release upload --clobber`).
  Do this only before anyone has installed it; otherwise release a new patch version.

## Permissions

Only the `Publish` job can write (`contents: write`, which creating a release needs), and only to
this repository; the packaging jobs read. The workflow runs only for tags pushed to this repository,
never for a pull request or in a fork, and pushing a tag needs write access. To restrict releases
further, add a tag ruleset (**Settings > Rules > Rulesets**) for the pattern `*-v*` that lets only
maintainers create, update or delete such tags.
