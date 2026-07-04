---
name: shadertoy-api
description: Fetching shaders from the Shadertoy.com REST API — Cloudflare TLS-fingerprint workaround (curl_cffi), API key, monthly budget, caches, and what is permanently blocked (/media/). Use when downloading shaders, debugging 403 errors from shadertoy.com, adding fetch code, or reasoning about API budget for the mass-test campaign.
---

# Shadertoy API access

## The one thing you must know first

shadertoy.com sits behind **Cloudflare bot protection** (since ~June 2026). Two
different mechanisms, two different outcomes:

1. **REST API** (`https://www.shadertoy.com/api/v1/...`) blocks by **TLS
   fingerprint (JA3)**, not by cookies or a JS challenge. Plain `requests`/`curl`
   gets HTTP 403 `"Bad request"` no matter what headers you send.
   **Working fix (already implemented):** `curl_cffi` with
   `impersonate="chrome"`. See `houdini/scripts/python/hshadertoy/api/shadertoy.py`
   (top of file) — it falls back to plain `requests` if curl_cffi is missing.
   - **Gotcha:** do NOT set a custom `User-Agent` when impersonating. Chrome TLS
     fingerprint + non-Chrome UA = 403 again.
   - **Residual 403s still happen occasionally even with impersonation** (1 in
     ~1000 campaign fetches: `XsXfR4`). Distinguish by message: the client's
     explicit `"Blocked by Cloudflare bot protection..."` text means curl_cffi
     is missing/inactive; a generic `"Shadertoy API request failed (HTTP 403)"`
     means impersonation WAS active and Cloudflare rejected once anyway —
     that's a normal retryable `ERROR` in the campaign ledger (just re-fetch
     later), NOT a reason to reinstall curl_cffi or escalate to Plan B.
   - `curl_cffi==0.15.0` is pinned in `requirements.txt`. For Houdini's own
     Python: `"C:/Program Files/Side Effects Software/Houdini 21.0.440/python311/python.exe" -m pip install --user curl_cffi`.
2. **Media assets** (`/media/...` textures, cubemaps, and the homepage) are behind
   the **full JavaScript managed challenge**. No impersonation target works.
   **Do not try to download media headlessly.** The project doesn't need to:
   the HDA builder maps Shadertoy asset IDs to bundled local assets via
   `houdini/scripts/python/hshadertoy/builder/hda/assets.json`.
   `download_media_file` / `download_shader_icon` in the API client are
   broken-but-unused.
3. **Plan B** if Cloudflare ever escalates onto the API path: Houdini 21's
   PySide6 ships `QtWebEngineWidgets` — fetch inside a `QWebEngineView` and read
   the JSON out of the page. Do NOT try to extract its cookies for `requests`
   (the TLS fingerprint still won't match).

## API key and limits

- App key: `NdHjRR` (hardcoded at `tests/campaign/campaign.py:75` and used by the
  API client). Keys are per-app, obtained at https://www.shadertoy.com/howto#q2.
- **Budget: 1500 API calls/month.** There is NO quota-query endpoint. The
  campaign tracks its own tally in `tests/campaign/api_budget.json`. The
  authoritative "you're out" signal is the API responding
  `{"Error":"Too many requests"}` — `campaign.py fetch` detects this, stops
  immediately, and leaves the current id unfetched (safe to re-run next month).
- Only shaders published as **Public + API** are reachable. The API returns
  "Shader not found" for BOTH genuinely missing ids and ids not set Public+API —
  you cannot distinguish them.
- Quick connectivity test shader: `lsGSDG`.

## Where fetched shaders live (check cache before spending API calls)

- `tests/campaign/cache/<id>.json` — campaign cache, the **inner** shader dict
  (`{"ver":…, "info":…, "renderpass":[…]}`).
- `tests/shaders/api/json/` and `tests/shaders/api/json_sp/` — older Phase-2
  caches; `campaign.py fetch` checks these first (cache hits cost 0 API calls).
- The full **shader index** `shader_all.json` (41,854 ids, newest→oldest) is what
  campaign sampling is built from — never re-fetch it casually; it's a huge call.

**Always prefer `python tests/campaign/campaign.py fetch --ids <id>` over ad-hoc
scripts** — it is cache-first, budget-tracked, and rate-limit-safe.

## JSON shape gotcha

The raw API response is `{"Shader": {inner}}`. The campaign cache stores only
`{inner}`. Houdini-side tools (e.g. `builder_test_headless.py`) expect the
**outer** `{"Shader": {...}}` shape — wrap cache files before feeding them to
Houdini tooling (see the houdini-testing skill).

## Related

- Mass-test fetch stage: `.claude/skills/mass-test-campaign/SKILL.md`
- Rendering originals headlessly (wgpu-shadertoy, same Cloudflare rules apply to
  its API module): `.claude/skills/render-compare/SKILL.md`
