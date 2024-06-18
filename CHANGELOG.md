# CHANGELOG

## v0.11.0 (2024-06-18)

### Feature

* feat: Async tokenizer (#86)

* feat: support async, wip

* feat: fix and add tests, examples, update readme

* fix: poetry lock

* fix: anyio -&gt; aiofiles

* fix: try 3.8

* fix: remove 3.7 from tests

* fix: poetry lock

* fix: add 3.7 back

* fix: poetry lock

* fix: poetry.lock

* ci: pipenv

* fix: pipenv

* fix: pipenv

* fix: pyproject

* fix: lock

* fix: version

* fix: Removed aiofiles

* ci: update python version,

* fix: switch from aiofiles to anyio, remove redundant comments

* chore: poetry lock

* fix: disable initializing async classes directly, cr comments

* test: fix import

* ci: add asyncio-mode to test workflow

* fix: to_thread -&gt; run_in_executor

* ci: add asyncio

* fix: cr comments

* fix: cr comments

---------

Co-authored-by: asafg &lt;asafg@ai21.com&gt; ([`3006cda`](https://github.com/AI21Labs/ai21-tokenizer/commit/3006cda2305a5ad31d705d3f6e3fe5605820e2f5))

## v0.10.0 (2024-06-16)

### Chore

* chore(release): v0.10.0 [skip ci] ([`1178ba7`](https://github.com/AI21Labs/ai21-tokenizer/commit/1178ba7fd1d980681cf143292848dacd51cdc2d5))

### Feature

* feat: remove python 3.7 support (#87) ([`58482ab`](https://github.com/AI21Labs/ai21-tokenizer/commit/58482abbda7767055345274b38c58ba9fd33b42f))

## v0.9.1 (2024-05-14)

### Chore

* chore(release): v0.9.1 [skip ci] ([`a260d76`](https://github.com/AI21Labs/ai21-tokenizer/commit/a260d76aece9992f03ac4f004edf6a6dd83f5f13))

### Fix

* fix: depend on less restrictive version of tokenizers (#85) ([`eab6a14`](https://github.com/AI21Labs/ai21-tokenizer/commit/eab6a14a6a788efbff708df0cded59d2392ef7f2))

## v0.9.0 (2024-03-28)

### Chore

* chore(release): v0.9.0 [skip ci] ([`974df9e`](https://github.com/AI21Labs/ai21-tokenizer/commit/974df9e8e9c46f5545dfb1a2603d51bac0eab750))

### Feature

* feat: Jamba instruct tokenizer (#84) ([`88ff9af`](https://github.com/AI21Labs/ai21-tokenizer/commit/88ff9aff504caa8928d68bcde4430c24fbbc23f1))

## v0.8.2 (2024-03-11)

### Chore

* chore(release): v0.8.2 [skip ci] ([`1146741`](https://github.com/AI21Labs/ai21-tokenizer/commit/11467416dd263b824f6a8711983eb5588fb037dc))

### Ci

* ci: add Python 3.12 to test matrix (#82)

* ci: add Python 3.12 to test matrix

* chore: use sentencepiece 0.2.0 or higher

* fix: update poetry.lock ([`8084117`](https://github.com/AI21Labs/ai21-tokenizer/commit/8084117c74813a99b79ecefd12888817470e1838))

### Fix

* fix: docs (#83) ([`c26949a`](https://github.com/AI21Labs/ai21-tokenizer/commit/c26949a62d5e612a7ff8132c6e6896b263be7b28))

### Unknown

* Update issue templates ([`86ea6e7`](https://github.com/AI21Labs/ai21-tokenizer/commit/86ea6e79a5670c0e8049ac587ed1b5f4b8790ae9))

## v0.8.1 (2024-01-07)

### Chore

* chore(release): v0.8.1 [skip ci] ([`fcacbf8`](https://github.com/AI21Labs/ai21-tokenizer/commit/fcacbf89a590e47d6ac3b8d385c9a6628a3ef4b2))

### Fix

* fix: re-ordered parameters in ctor to avoid a breaking change (#79) ([`6c1b608`](https://github.com/AI21Labs/ai21-tokenizer/commit/6c1b6088c0914ffc77b53613047606c398e0557c))

## v0.8.0 (2024-01-03)

### Chore

* chore(release): v0.8.0 [skip ci] ([`c8b54df`](https://github.com/AI21Labs/ai21-tokenizer/commit/c8b54dff67c13587943f03198ec5a4e1dca7be88))

* chore(deps-dev): bump pytest from 7.2.1 to 7.4.4 (#75)

Bumps [pytest](https://github.com/pytest-dev/pytest) from 7.2.1 to 7.4.4.
- [Release notes](https://github.com/pytest-dev/pytest/releases)
- [Changelog](https://github.com/pytest-dev/pytest/blob/main/CHANGELOG.rst)
- [Commits](https://github.com/pytest-dev/pytest/compare/7.2.1...7.4.4)

---
updated-dependencies:
- dependency-name: pytest
  dependency-type: direct:development
  update-type: version-update:semver-minor
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt;
Co-authored-by: asafgardin &lt;147075902+asafgardin@users.noreply.github.com&gt; ([`081dda3`](https://github.com/AI21Labs/ai21-tokenizer/commit/081dda305ebc33af78ad433d511bdef3d63e1307))

### Feature

* feat: Add start_of_line to decode (#77)

* feat: Add start_of_line param to decode

* test: added unittest with start_of_line=True and False ([`182a8d1`](https://github.com/AI21Labs/ai21-tokenizer/commit/182a8d10020862c233f7f67cddb965eee2398b98))

## v0.7.0 (2024-01-02)

### Chore

* chore(release): v0.7.0 [skip ci] ([`26f34b2`](https://github.com/AI21Labs/ai21-tokenizer/commit/26f34b290cdc6b5166872bd6b8af5ca53d736936))

### Feature

* feat: Init tokenizer from filehandle (#76)

* feat: allow creating JurassicTokenizer from model file handle

* fix: Add default for model_path and model_file_handle

* feat: Add JurassicTokenizer.from_file_path classmethod

* fix: remove model_path=None in JurassicTokenizer.from_file_handle

* fix: rename _assert_exactly_one to _validate_init and make it not static

* refactor: semantics

* test: Added tests

---------

Co-authored-by: Asaf Gardin &lt;asafg@ai21.com&gt; ([`dcb73a7`](https://github.com/AI21Labs/ai21-tokenizer/commit/dcb73a72348e576b06cd4a066e06141ceae37a44))

## v0.6.0 (2023-12-28)

### Chore

* chore(release): v0.6.0 [skip ci] ([`7b8348d`](https://github.com/AI21Labs/ai21-tokenizer/commit/7b8348d303eb54c4a75ca1c58be5c08c35ec3de8))

* chore: add test case for encode with is_start=False (#74)

* chore: add test case for encode with is_start=False

* fix: split is_start=False to a different testcase ([`77c0a39`](https://github.com/AI21Labs/ai21-tokenizer/commit/77c0a39d1bcde81cc0166a512eb454dad6d3c569))

### Feature

* feat: Add decode with offsets (#73)

* feat: Add decode_with_offsets() to JurassicTokenizer

* refactor: remove kwargs from decode_with_offsets since it&#39;s not used

* chore: Add unittest for decode and for offsets

* fix: test only decode_with_offsets

* fix: dummy for returned offsets in decode_with_offsets ([`a5a7bb4`](https://github.com/AI21Labs/ai21-tokenizer/commit/a5a7bb4b27fa4f74a0b1a1d6874599556a35c1c5))

* feat: Add the is_start parameter to JurassicTokenizer.encode() (#72)

* feat: Add the is_start parameter to JurassicTokenizer.encode()

* refactor: take &#39;is_start&#39; from kwargs ([`296bda5`](https://github.com/AI21Labs/ai21-tokenizer/commit/296bda5578edd57ff58d6763b3ccd5b9ba709795))

## v0.5.0 (2023-12-28)

### Chore

* chore(release): v0.5.0 [skip ci] ([`96f384f`](https://github.com/AI21Labs/ai21-tokenizer/commit/96f384f2873ec363016c8def483f59ac535acba9))

### Feature

* feat: Add more special tokens (#71)

* fix: commitizen tag starts with &#34;v&#34;

* feat: add eos_id

* feat: Add newline_id

* fix: typo &#34;_newline_piece&#34; instead of &#34;newline_piece&#34;

* fix: newline_id already existed as &#34;private&#34;. Just make it &#34;public&#34;

* fix: forgot to rename everywhere ([`9a9e1a8`](https://github.com/AI21Labs/ai21-tokenizer/commit/9a9e1a8a21dbfd4d3f983ce8ec1f97335470e2c6))

### Fix

* fix: commitizen tag starts with &#34;v&#34; (#70) ([`cf495ad`](https://github.com/AI21Labs/ai21-tokenizer/commit/cf495ada8341131c0d022f02d2c9e86cee8723ba))

## v0.4.0 (2023-12-28)

### Chore

* chore(release): v0.4.0 [skip ci] ([`b761edc`](https://github.com/AI21Labs/ai21-tokenizer/commit/b761edc0947b132192d4f963dd8d8704df4ebac4))

### Feature

* feat: add pad_id and bos_id to jurassic_tokenizer (#69) ([`ffb2ce3`](https://github.com/AI21Labs/ai21-tokenizer/commit/ffb2ce38aa59eec03305c4fdc6bfedd99c7b5255))

## v0.3.11 (2023-12-27)

### Chore

* chore(release): v0.3.11 [skip ci] ([`5280149`](https://github.com/AI21Labs/ai21-tokenizer/commit/528014946c5d35bbd9a5c8fa8c6eed7754e1f2f1))

### Fix

* fix: BaseTokenizer in init (#68) ([`3cc71e7`](https://github.com/AI21Labs/ai21-tokenizer/commit/3cc71e747a0b5d5450dbe70618456d415171ae23))

## v0.3.10 (2023-12-27)

### Chore

* chore(release): v0.3.10 [skip ci] ([`1601535`](https://github.com/AI21Labs/ai21-tokenizer/commit/16015359022592a058a48893076fd4c34ab85b74))

* chore(deps-dev): bump safety from 2.3.4 to 2.3.5 (#64)

Bumps [safety](https://github.com/pyupio/safety) from 2.3.4 to 2.3.5.
- [Release notes](https://github.com/pyupio/safety/releases)
- [Changelog](https://github.com/pyupio/safety/blob/main/CHANGELOG.md)
- [Commits](https://github.com/pyupio/safety/compare/2.3.4...2.3.5)

---
updated-dependencies:
- dependency-name: safety
  dependency-type: direct:development
  update-type: version-update:semver-patch
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt; ([`95696bb`](https://github.com/AI21Labs/ai21-tokenizer/commit/95696bb2eee8e3ddfff3d4e297ba8d8633c7c112))

* chore(deps-dev): bump ruff from 0.0.285 to 0.1.8 (#63)

Bumps [ruff](https://github.com/astral-sh/ruff) from 0.0.285 to 0.1.8.
- [Release notes](https://github.com/astral-sh/ruff/releases)
- [Changelog](https://github.com/astral-sh/ruff/blob/main/CHANGELOG.md)
- [Commits](https://github.com/astral-sh/ruff/compare/v0.0.285...v0.1.8)

---
updated-dependencies:
- dependency-name: ruff
  dependency-type: direct:development
  update-type: version-update:semver-minor
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt; ([`81123d3`](https://github.com/AI21Labs/ai21-tokenizer/commit/81123d3a7afc9794dc21657cc5bab9dc94a123b2))

* chore(deps-dev): bump black from 22.12.0 to 23.3.0 (#61)

Bumps [black](https://github.com/psf/black) from 22.12.0 to 23.3.0.
- [Release notes](https://github.com/psf/black/releases)
- [Changelog](https://github.com/psf/black/blob/main/CHANGES.md)
- [Commits](https://github.com/psf/black/compare/22.12.0...23.3.0)

---
updated-dependencies:
- dependency-name: black
  dependency-type: direct:development
  update-type: version-update:semver-major
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt; ([`7190d28`](https://github.com/AI21Labs/ai21-tokenizer/commit/7190d28ab2ee7d8a5bfcf8b1b90606940f4a25ad))

* chore(deps-dev): bump safety from 2.3.4 to 2.3.5 (#60)

Bumps [safety](https://github.com/pyupio/safety) from 2.3.4 to 2.3.5.
- [Release notes](https://github.com/pyupio/safety/releases)
- [Changelog](https://github.com/pyupio/safety/blob/main/CHANGELOG.md)
- [Commits](https://github.com/pyupio/safety/compare/2.3.4...2.3.5)

---
updated-dependencies:
- dependency-name: safety
  dependency-type: direct:development
  update-type: version-update:semver-patch
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt; ([`2fa7bef`](https://github.com/AI21Labs/ai21-tokenizer/commit/2fa7bef0b26c891778180970f66932b42126c8a1))

### Fix

* fix: JurassicTokenizer in init (#67) ([`253ae07`](https://github.com/AI21Labs/ai21-tokenizer/commit/253ae073ba20ebef544cde803c9180975aa70a5d))

### Refactor

* refactor: Added __all__ in __init__ (#65)

* refactor: Added __all__ in __init__

* fix: tests

* refactor: added __version__ to __all__ ([`c0d9286`](https://github.com/AI21Labs/ai21-tokenizer/commit/c0d9286a2a8d8ae7d7cb418620d709be7b7d8193))

* refactor: sentencepiece version to support all patch versions (#66) ([`845008c`](https://github.com/AI21Labs/ai21-tokenizer/commit/845008cf61f86b40516bbc1836e0b4cfd0559192))

## v0.3.9 (2023-11-27)

### Chore

* chore(release): v0.3.9 [skip ci] ([`84f17da`](https://github.com/AI21Labs/ai21-tokenizer/commit/84f17dafee88962ead47d19a1ea2eb3998de3c18))

* chore: add github badges (#58) ([`821455c`](https://github.com/AI21Labs/ai21-tokenizer/commit/821455c6bf2e897a3bf3619d2e078e2daf3d4653))

* chore(deps-dev): bump urllib3 from 2.0.4 to 2.0.7 (#57)

Bumps [urllib3](https://github.com/urllib3/urllib3) from 2.0.4 to 2.0.7.
- [Release notes](https://github.com/urllib3/urllib3/releases)
- [Changelog](https://github.com/urllib3/urllib3/blob/main/CHANGES.rst)
- [Commits](https://github.com/urllib3/urllib3/compare/2.0.4...2.0.7)

---
updated-dependencies:
- dependency-name: urllib3
  dependency-type: indirect
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt; ([`93ef6d6`](https://github.com/AI21Labs/ai21-tokenizer/commit/93ef6d6a18ca7ce7fb65fdf4cbc09e7105bcd8c3))

### Fix

* fix: Modify badges (#59)

* docs: fixed url

* fix: inline

* fix: README.md ([`20b7090`](https://github.com/AI21Labs/ai21-tokenizer/commit/20b709012059719a5084b7abed745f8705f0e8a1))

## v0.3.8 (2023-11-26)

### Chore

* chore(release): v0.3.8 [skip ci] ([`645ff5e`](https://github.com/AI21Labs/ai21-tokenizer/commit/645ff5ea0ffda722dae0fd211bded29f75b72066))

### Fix

* fix: readme example (#56) ([`b713da8`](https://github.com/AI21Labs/ai21-tokenizer/commit/b713da8cdfe6434ac03cd25eaea7036ed172bf59))

## v0.3.7 (2023-11-23)

### Chore

* chore(release): v0.3.7 [skip ci] ([`a181ae6`](https://github.com/AI21Labs/ai21-tokenizer/commit/a181ae67bee062b1ac74633a0b6d37fc2a34c4fc))

### Ci

* ci: workflow dispatch for release (#54) ([`dbf5609`](https://github.com/AI21Labs/ai21-tokenizer/commit/dbf5609f63966804ea56e3c6b3e671498cd9feb3))

* ci: Automate pypi publish (#53)

* ci: Automate pypi publish on new release

* fix: Remove comment

* fix: title of action

* fix: title of action ([`7c04fda`](https://github.com/AI21Labs/ai21-tokenizer/commit/7c04fdab10f57c4ed61ce974a5f98ed2b5ec0abd))

### Fix

* fix: Examples in readme (#55)

* ci: workflow dispatch for release

* docs: Updated readme with more examples

* docs: Added docs to base class ([`94f3a3c`](https://github.com/AI21Labs/ai21-tokenizer/commit/94f3a3c9539346ac1bc501f6c83ce1e3525d055a))

## v0.3.6 (2023-11-22)

### Chore

* chore(release): v0.3.6 [skip ci] ([`550644d`](https://github.com/AI21Labs/ai21-tokenizer/commit/550644d181a36eb9a2d4312959a27a04743cde25))

### Ci

* ci: added python version to pypi publish (#50) ([`83cbbea`](https://github.com/AI21Labs/ai21-tokenizer/commit/83cbbeaa2626ff0adfc10a4d325a345322fade70))

* ci: exclude changelog (#48) ([`9222c55`](https://github.com/AI21Labs/ai21-tokenizer/commit/9222c5566b8d220ca70e9c1eb6ed6aec679c9873))

### Documentation

* docs: CODEOWNERS (#49) ([`1e6513b`](https://github.com/AI21Labs/ai21-tokenizer/commit/1e6513badd83cde1b48da6dcf5c31ae0ec1795c3))

### Fix

* fix: support ai21_tokenizer.__version__ (#52) ([`13944eb`](https://github.com/AI21Labs/ai21-tokenizer/commit/13944eb7a8ae540e0be329abb12920e374c62b23))

## v0.3.5 (2023-11-22)

### Chore

* chore(release): v0.3.5 [skip ci] ([`217b14e`](https://github.com/AI21Labs/ai21-tokenizer/commit/217b14ef23e7e63da9e0de5896bb002dcf89697d))

### Fix

* fix: newline n and prettier (#47)

* fix: newline n and prettier

* fix: exclude

* fix: exclude from pretty ([`f5c9204`](https://github.com/AI21Labs/ai21-tokenizer/commit/f5c920422c31b671d09af43758560dd2779ac358))

## v0.3.4 (2023-11-22)

### Chore

* chore(release): v0.3.4 [skip ci] ([`be67c79`](https://github.com/AI21Labs/ai21-tokenizer/commit/be67c79136f6d1c284cb4f373d39fb0ef903733a))

### Fix

* fix: newline r (#46) ([`bc97ae1`](https://github.com/AI21Labs/ai21-tokenizer/commit/bc97ae14d403ee6e4313a1b313ea328dfe265a8a))

## v0.3.3 (2023-11-22)

### Chore

* chore(release): v0.3.3 [skip ci] ([`f6db520`](https://github.com/AI21Labs/ai21-tokenizer/commit/f6db5207bcf415f1b89bb3270fc1a7dbe6dc8031))

### Ci

* ci: on push release (#43) ([`95bfc95`](https://github.com/AI21Labs/ai21-tokenizer/commit/95bfc951906f79b0cdfea7544b4bb49222ae19a6))

### Documentation

* docs: readme (#44) ([`af23ac4`](https://github.com/AI21Labs/ai21-tokenizer/commit/af23ac4442cb358809aa27d4cfd2fecb95060044))

### Fix

* fix: setup (#45) ([`1b8c00f`](https://github.com/AI21Labs/ai21-tokenizer/commit/1b8c00f3240e94237f084202fd77ee8c7ef5a726))

## v0.3.2 (2023-11-21)

### Chore

* chore(release): v0.3.2 [skip ci] ([`f3b2e73`](https://github.com/AI21Labs/ai21-tokenizer/commit/f3b2e737c0dbf8c29524e83ffdebe95d3a12c1de))

### Ci

* ci: Added newline sequence (#41) ([`63a1898`](https://github.com/AI21Labs/ai21-tokenizer/commit/63a1898f49cd6d60bf2fbf98ea62915c55b91cb6))

### Fix

* fix: Version path (#42)

* fix: version

* fix: version variables

* fix: name ([`ee4d744`](https://github.com/AI21Labs/ai21-tokenizer/commit/ee4d744bc8a5565f91e888b1ab81d9fd7011c0a2))

## v0.3.1 (2023-11-21)

### Chore

* chore(release): v0.3.1 [skip ci] ([`2eebeb8`](https://github.com/AI21Labs/ai21-tokenizer/commit/2eebeb8418bd7a915eb2a82142dc73962b4a5fbb))

### Fix

* fix: test 2 (#40) ([`2028613`](https://github.com/AI21Labs/ai21-tokenizer/commit/20286135bbdc5f3193bdf378ab000145d95c3c2c))

* fix: Test bump 1 (#39)

* fix: crlf forbid

* fix: test 1 ([`26da29e`](https://github.com/AI21Labs/ai21-tokenizer/commit/26da29e921dd8dd546a127c76214f20a0718f17d))

## v0.3.0 (2023-11-21)

### Chore

* chore(release): v0.3.0 [skip ci] ([`0c4ada0`](https://github.com/AI21Labs/ai21-tokenizer/commit/0c4ada02851933763da28bdeb726380930b7fb46))

### Documentation

* docs: Release md update before publish (#36)

* fix: Added support for both str and path

* fix: rename package

* fix: updated pre commits and added new one

* docs: Updated docs

* ci: down grade

* docs: Added another example ([`18ccbeb`](https://github.com/AI21Labs/ai21-tokenizer/commit/18ccbeb89745491fa2d5ac92b8e017ee2af4ca88))

* docs: CONTRIBUTING.md (#35)

* docs: CONTRIBUTING.md

* ci: end_of_line fix

* docs: inv test ([`e282440`](https://github.com/AI21Labs/ai21-tokenizer/commit/e2824402e375aa4e4714fb1afa3c212abd276c2f))

### Feature

* feat: Added char for testing (#37) ([`40d3feb`](https://github.com/AI21Labs/ai21-tokenizer/commit/40d3febf9a4df9f54b91e50981326b71ce362c89))

### Fix

* fix: string example (#38) ([`833038c`](https://github.com/AI21Labs/ai21-tokenizer/commit/833038c0ff348cfa240764346e2faffea09ed6ac))

## v0.2.0 (2023-11-21)

### Chore

* chore(release): v0.2.0 [skip ci] ([`8988faa`](https://github.com/AI21Labs/ai21-tokenizer/commit/8988faa1419069916aedcb317e7c813a57a7d03d))

* chore(deps-dev): bump black from 22.12.0 to 23.3.0 (#32)

Bumps [black](https://github.com/psf/black) from 22.12.0 to 23.3.0.
- [Release notes](https://github.com/psf/black/releases)
- [Changelog](https://github.com/psf/black/blob/main/CHANGES.md)
- [Commits](https://github.com/psf/black/compare/22.12.0...23.3.0)

---
updated-dependencies:
- dependency-name: black
  dependency-type: direct:development
  update-type: version-update:semver-major
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt;
Co-authored-by: asafgardin &lt;147075902+asafgardin@users.noreply.github.com&gt; ([`bb4986e`](https://github.com/AI21Labs/ai21-tokenizer/commit/bb4986e880f24eebc7be7eee050d85c4d56a1aa9))

### Feature

* feat: Tokenizer factory (#31)

* feat: Added tokenizer abc and factory

* fix: api to receive default and none

* fix: example

* fix: factory and tests

* fix: rename base

* fix: rename base class

* fix: rename package

* fix: example

* fix: readme and tasks

* docs: factory class

* docs: renames

* fix: directory hierarchy in tests

* fix: rename package

* chore(release): v0.1.2 [skip ci]

* fix: rename package

* ci: example

* fix: assert in example

* fix: src_path

---------

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`e55cd1d`](https://github.com/AI21Labs/ai21-tokenizer/commit/e55cd1dac5ad501a0a48c369a28d061f37950f5f))

### Fix

* fix: token name (#34) ([`2b229b2`](https://github.com/AI21Labs/ai21-tokenizer/commit/2b229b28ace8ac72dcc9bc727d187350020a2e12))

## v0.1.2 (2023-11-21)

### Chore

* chore(release): v0.1.2 [skip ci] ([`5b1dc14`](https://github.com/AI21Labs/ai21-tokenizer/commit/5b1dc140213615bbe8a6a65caea8eea2ec7d3cbc))

* chore(deps-dev): bump safety from 2.3.4 to 2.3.5 (#28)

Bumps [safety](https://github.com/pyupio/safety) from 2.3.4 to 2.3.5.
- [Release notes](https://github.com/pyupio/safety/releases)
- [Changelog](https://github.com/pyupio/safety/blob/main/CHANGELOG.md)
- [Commits](https://github.com/pyupio/safety/compare/2.3.4...2.3.5)

---
updated-dependencies:
- dependency-name: safety
  dependency-type: direct:development
  update-type: version-update:semver-patch
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt; ([`28118ac`](https://github.com/AI21Labs/ai21-tokenizer/commit/28118acd7477a128eecf36119d60d50fa1fbb169))

* chore(deps-dev): bump pytest-mock from 3.10.0 to 3.11.1 (#24)

Bumps [pytest-mock](https://github.com/pytest-dev/pytest-mock) from 3.10.0 to 3.11.1.
- [Release notes](https://github.com/pytest-dev/pytest-mock/releases)
- [Changelog](https://github.com/pytest-dev/pytest-mock/blob/main/CHANGELOG.rst)
- [Commits](https://github.com/pytest-dev/pytest-mock/compare/v3.10.0...v3.11.1)

---
updated-dependencies:
- dependency-name: pytest-mock
  dependency-type: direct:development
  update-type: version-update:semver-minor
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt;
Co-authored-by: dependabot[bot] &lt;49699333+dependabot[bot]@users.noreply.github.com&gt;
Co-authored-by: asafgardin &lt;147075902+asafgardin@users.noreply.github.com&gt; ([`0a36f21`](https://github.com/AI21Labs/ai21-tokenizer/commit/0a36f21e0c375b12f743e4c89c90b5391502d2b3))

### Ci

* ci: Remove install from publish (#29)

* ci: Removed install dependency

* ci: changlog changes ([`f8e8392`](https://github.com/AI21Labs/ai21-tokenizer/commit/f8e8392c717d93d1a103d1eec84b7580beaa1b14))

* ci: dependabot pr limit (#27)

* ci: dependabot pr limit

* ci: dependabot pr limit ([`9b2c4f8`](https://github.com/AI21Labs/ai21-tokenizer/commit/9b2c4f885a5f2be3076f3c4563de33efb3f7af9a))

### Fix

* fix: workflow dispatch for release action (#33) ([`f81b4ab`](https://github.com/AI21Labs/ai21-tokenizer/commit/f81b4abbe4fe45163dec45dfce89677fe673e46b))

## v0.1.1 (2023-11-20)

### Chore

* chore(release): v0.1.1 [skip ci] ([`f5150f5`](https://github.com/AI21Labs/ai21-tokenizer/commit/f5150f5473bb95759cf389db2033789f69e3ac38))

### Fix

* fix: used PAT (#26)

* test: write to main

* fix: token

* test: debug

* test: debugging tokens

* test: uncomment

* test: write to main

* fix: token

* test: debug

* test: debugging tokens

* test: uncomment

* fix: Changed to main ([`2ff12c9`](https://github.com/AI21Labs/ai21-tokenizer/commit/2ff12c9b4d490eec0d089b18a6bdabda7e23e952))

## v0.1.0 (2023-11-20)

### Chore

* chore(release): v0.1.0 [skip ci] ([`48607f9`](https://github.com/AI21Labs/ai21-tokenizer/commit/48607f92d977ce4d725879d5ac1306c8efd3255c))

### Ci

* ci: Create dependabot.yml (#19)

* ci: Create dependabot.yml

* fix: commit-message prefix

* fix: Added more config to dependabot action ([`23faaa8`](https://github.com/AI21Labs/ai21-tokenizer/commit/23faaa81d37a673d56542b367470aab6013d2b39))

### Feature

* feat: Pypi publish (#18)

* feast: Added setup.py

* feast: Added publish.yaml ([`77ee751`](https://github.com/AI21Labs/ai21-tokenizer/commit/77ee751b6d7610c4b4955ad488579a88f47d4f8b))

* feat: test PAT (#9) ([`a10b6a4`](https://github.com/AI21Labs/ai21-tokenizer/commit/a10b6a494c20e8929dc3ba355527f8290f27afbf))

* feat: Added semantic prs actions (#8) ([`afab5ff`](https://github.com/AI21Labs/ai21-tokenizer/commit/afab5ff05154da698120d67f7a018907c6ed1ecc))

### Fix

* fix: Added permissions (#25)

* fix: Added permissions

* fix: permissions location

* fix: verbose

* fix: Removed bad input &#34;root_options&#34; ([`d684575`](https://github.com/AI21Labs/ai21-tokenizer/commit/d684575675e85e70af96639cc3c33f125770e7da))

* fix: Change token (#17)

* fix: token key

* fix: token github

* fix: token github cls

* fix: token github cls ([`ec4f35b`](https://github.com/AI21Labs/ai21-tokenizer/commit/ec4f35bf477a602e648f1d6d212c2a388b9c97e0))

* fix: Change token (#16)

* fix: token key

* fix: token github ([`d876a43`](https://github.com/AI21Labs/ai21-tokenizer/commit/d876a43b4bde91f202f93fb22411df088af49293))

* fix: token key (#15) ([`0b76344`](https://github.com/AI21Labs/ai21-tokenizer/commit/0b76344bfcad9c03cc50d19ad2ecdf5d2127b3e2))

* fix: keys (#14) ([`b064ea6`](https://github.com/AI21Labs/ai21-tokenizer/commit/b064ea657945e7e3201d9b7678e2551ad0c75f89))

* fix: Test token (#11)

* feat: test PAT

* feat: test github token

* fix: PAT ([`94f64b6`](https://github.com/AI21Labs/ai21-tokenizer/commit/94f64b6bded3bba3e6f616cfd37315bacf68b3f4))

* fix: Test token (#10)

* feat: test PAT

* feat: test github token ([`52484fe`](https://github.com/AI21Labs/ai21-tokenizer/commit/52484fe2a0951652ed4d34c17cf48ca47669ab11))

* fix: root_options verbose (#6) ([`220ba5b`](https://github.com/AI21Labs/ai21-tokenizer/commit/220ba5b47e9f2c4b3db0c38291abe4a983b35c2e))

* fix: Release action test (#5)

* fix: Added release step

* fix: branch name for testing

* fix: Removed comment

* chore(release): v0.0.1 [skip ci]

* fix: branch rename

* fix: Removed CHANGELOG.md

---------

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`fae5423`](https://github.com/AI21Labs/ai21-tokenizer/commit/fae5423193754d4e60826e1a74dfb70e6e463c47))

### Test

* test: Test token (#13)

* feat: test PAT

* feat: test github token

* fix: PAT

* test: Added test step

* test: Added test step

* test: Added token to use ([`d445e36`](https://github.com/AI21Labs/ai21-tokenizer/commit/d445e3688412ca4142c9309b28de6400a1f54220))

* test: Test token (#12)

* feat: test PAT

* feat: test github token

* fix: PAT

* test: Added test step

* test: Added test step ([`9619c3e`](https://github.com/AI21Labs/ai21-tokenizer/commit/9619c3e064e0014d2dbf776a7bb18cd7a914c817))

### Unknown

* Add kwargs to functions (#7)

* feat: added kwargs

* test: Added tests ([`efecff9`](https://github.com/AI21Labs/ai21-tokenizer/commit/efecff97f0291a0f155208ba09c83dfdcfd9247c))

* Release action (#4)

* feat: Added release action

* fix: Removed unnecessary code

* fix: testing on branch

* fix: removed node install

* fix: Removed unnecessary step

* fix: base-branch

* fix: removed code

* 0.0.2

* feat: python-semantic-release test

* fix: branch name

* fix: branch name in .toml

* fix: change from branch to match

* fix: Added release_action part

* chore(release): v0.1.0 [skip ci]

* refactor: removed CHANGELOG.md

* fix: branch to main

* feat: Added version.py to version_variable

* feat: Upgraded python-semantic-release

* feat: Added python-semantic-release

* fix: Removed unnecessary file

* fix: Changed version

---------

Co-authored-by: github-action &lt;41898282+github-actions[bot]@users.noreply.github.com&gt;
Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`694683a`](https://github.com/AI21Labs/ai21-tokenizer/commit/694683ac39330944368a1e6ca3370857804bb960))

* Add code (#2)

* feat: Jurassic tokenizer

* fix: remove is_start

* fix: add types

* fix: add types

* chore: extracted utils

* fix: simplified tokenizer even more

* fix: simplified tokenizer even more

* feat: Added tests

* feat: exposed prop

---------

Co-authored-by: Asaf Gardin &lt;asafg@ai21.com&gt; ([`6b80a05`](https://github.com/AI21Labs/ai21-tokenizer/commit/6b80a05549267e44e59f8ae40a92e4de68df979c))

* First commit (#1)

* feat: init project ([`f50565e`](https://github.com/AI21Labs/ai21-tokenizer/commit/f50565eeb8d7259dec565f1292592d66b479f962))

* Initial commit ([`556d3e6`](https://github.com/AI21Labs/ai21-tokenizer/commit/556d3e64af018cf7269d9face370099900aec8eb))
