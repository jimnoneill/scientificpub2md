# Publishing to PyPI

`scientificpub2md` is packaged for PyPI. Once published, anyone can:

```bash
pip install scientificpub2md
```

There are two ways to publish a release. **Trusted Publishing (recommended)** needs no API token.

---

## Option A — GitHub Release + Trusted Publishing (recommended)

One-time setup on PyPI (no token stored anywhere):

1. Create the project's trusted publisher at
   <https://pypi.org/manage/account/publishing/> → "Add a new pending publisher":
   - **PyPI Project Name:** `scientificpub2md`
   - **Owner:** `jimnoneill`
   - **Repository name:** `scientificpub2md`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
2. (Optional but recommended) In the GitHub repo: **Settings → Environments → New environment → `pypi`**.

Then, for every release:

```bash
# bump the version in pyproject.toml + scientificpub2md/__init__.py first, then:
git tag v0.2.0
git push origin v0.2.0
gh release create v0.2.0 --generate-notes
```

Publishing the GitHub Release triggers `.github/workflows/publish.yml`, which builds the
sdist + wheel, runs `twine check`, and uploads to PyPI via OIDC. Done.

---

## Option B — Manual upload with an API token

```bash
python -m pip install --upgrade build twine
python -m build                 # -> dist/scientificpub2md-<ver>.tar.gz + .whl
python -m twine check dist/*

# Get a token at https://pypi.org/manage/account/token/ and:
python -m twine upload dist/*   # username: __token__   password: <your pypi token>
```

Tip: test first against TestPyPI —
`python -m twine upload --repository testpypi dist/*`, then
`pip install --index-url https://test.pypi.org/simple/ scientificpub2md`.

---

## Release checklist

- [ ] Bump `version` in `pyproject.toml` **and** `scientificpub2md/__init__.py` (keep them in sync).
- [ ] `python -m build && python -m twine check dist/*` passes.
- [ ] Tag `vX.Y.Z` and publish the GitHub Release (Option A) or `twine upload` (Option B).
