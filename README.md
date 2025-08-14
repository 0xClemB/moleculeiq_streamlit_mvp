# MoleculeIQ v17 – Live PubChem + PatentsView (Offline-capable)
- PBKDF2 login (no bcrypt). Test users in TEST_ACCOUNTS.txt.
- Examples dropdown; live PubChem + PatentsView.
- Offline fallback using bundled mock datasets when OFFLINE_MODE=true or network fails.

## Run
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set OFFLINE_MODE=true for offline demo
streamlit run app.py
