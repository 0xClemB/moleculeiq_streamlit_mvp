
# MoleculeIQ – Streamlit MVP (v14, login-enabled)

This build adds a simple **username/password login** (bcrypt) with four test personas and keeps all report/linking fixes from v13.

## Test Accounts
See `TEST_ACCOUNTS.txt` for usernames and passwords:
- scientist / Chem!2025Test
- attorney  / IPLaw!2025OK
- founder   / Founder#2025
- tto       / TTOsecure#2025

## Run
```bash
unzip moleculeiq_streamlit_mvp_v14.zip
cd moleculeiq_streamlit_mvp_v14
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# or Git Bash:
source .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Passwords are hashed with bcrypt; plaintext examples are provided only in TEST_ACCOUNTS.txt for demo use.
- For production, replace this with **OIDC/SAML SSO** via a gateway as per our plan.


---

## v16 changes
- Switched login hashing from `bcrypt` to **PBKDF2-HMAC (stdlib)** — no external binary dependency.
- Replaced deprecated `st.experimental_rerun()` with `st.rerun()`.
- Kept v15 features: login personas, Examples dropdown, TOC links, claim diff legend, widened columns, verified patent links, etc.

### Deploying on Streamlit Cloud
No `bcrypt` needed. Just push to GitHub and deploy. If you pinned a runtime, Python 3.11+ is fine.
