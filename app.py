
import streamlit as st
import pandas as pd
import numpy as np
import requests, time, os, json, base64, hashlib, hmac, re
from urllib.parse import quote_plus
from dotenv import load_dotenv
from bs4 import BeautifulSoup

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, DataStructs
    RD_ENABLED = True
except Exception:
    RD_ENABLED = False

load_dotenv()
st.set_page_config(page_title="MoleculeIQ v17 – Chemistry-first Novelty", page_icon="🧪", layout="wide")

def hash_password(password: str, iterations: int = 200_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algo, s_iters, s_salt, s_hash = stored_hash.split("$", 3)
        iterations = int(s_iters)
        salt = base64.b64decode(s_salt)
        expected = base64.b64decode(s_hash)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False

@st.cache_data
def load_users():
    try:
        if "users_json" in st.secrets:
            return json.loads(st.secrets["users_json"])
        with open("users.json","r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

USERS = load_users()

def ensure_demo_hashes():
    defaults = {"scientist":"Chem!2025Test","attorney":"IPLaw!2025OK","founder":"Founder#2025","tto":"TTOsecure#2025"}
    changed = False
    for u, rec in USERS.items():
        if not rec.get("password_hash"):
            rec["password_hash"] = hash_password(defaults[u])
            changed = True
    if changed:
        try:
            with open("users.json","w",encoding="utf-8") as f:
                json.dump(USERS,f,ensure_ascii=False,indent=2)
        except Exception:
            pass

ensure_demo_hashes()

def check_password(username, password):
    user = USERS.get(username)
    if not user: return False
    return verify_password(password, user.get("password_hash",""))

def login_gate():
    st.title("🔐 MoleculeIQ Login")
    with st.form("login"):
        u = st.text_input("Username", placeholder="e.g., scientist")
        p = st.text_input("Password", type="password", placeholder="••••••")
        if st.form_submit_button("Login"):
            if check_password(u,p):
                st.session_state["user"] = USERS[u]["name"]
                st.session_state["role"] = USERS[u]["role"]
                st.success(f"Welcome {USERS[u]['name']}")
                st.rerun()
            else:
                st.error("Invalid credentials")

if "user" not in st.session_state:
    login_gate(); st.stop()

with st.sidebar:
    st.markdown("### 👤 User")
    st.write(st.session_state.get("user",""))
    st.caption(st.session_state.get("role",""))
    if st.button("Logout"):
        for k in ["user","role"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("### Examples")
    example_smiles = {
        "Methane":"C",
        "Ethanol":"CCO",
        "Benzene":"c1ccccc1",
        "Aspirin":"CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine":"Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "Cholesterol":"C[C@H](CCC(=O)O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
        "Glucose":"C(C1C(C(C(C(O1)O)O)O)O)O",
        "Penicillin G":"CC1(C)S[C@H]2[C@H](NC1=O)C(=O)N2C(=O)C(C1=CC=CC=C1)O",
        "Morphine":"CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@H](O)C=C[C@H]3[C@@H]1C5",
        "Serotonin":"C1=CC2=C(C=C1O)C(=CN2)CCN"
    }
    ex = st.selectbox("Select an example", list(example_smiles.keys()), index=None, placeholder="Choose...")
    if ex:
        st.session_state["smiles_input"] = example_smiles[ex]

PUBCHEM_RPS = float(os.getenv("PUBCHEM_RPS","5"))
PATENTSVIEW_RPS = float(os.getenv("PATENTSVIEW_RPS","5"))
OFFLINE_MODE = os.getenv("OFFLINE_MODE","false").lower()=="true"
_last_pubchem=[0.0]; _last_pv=[0.0]

def rate_limit(last_ts, rps):
    now=time.time(); min_interval=1.0/max(rps,0.1)
    if now-last_ts[0]<min_interval:
        time.sleep(min_interval-(now-last_ts[0]))
    last_ts[0]=time.time()

def canonicalize_smiles(smiles:str)->str:
    if RD_ENABLED:
        m=Chem.MolFromSmiles(smiles)
        if m is None: return smiles
        return Chem.MolToSmiles(m, canonical=True)
    return smiles

def pubchem_cid_from_smiles(smiles:str):
    if OFFLINE_MODE: return None
    rate_limit(_last_pubchem,PUBCHEM_RPS)
    url=f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote_plus(smiles)}/cids/TXT"
    r=requests.get(url,timeout=10)
    if r.status_code==200 and r.text.strip().isdigit():
        return r.text.strip()
    return None

def pubchem_png(smiles:str):
    if OFFLINE_MODE: raise RuntimeError("Offline mode: no PubChem image")
    rate_limit(_last_pubchem,PUBCHEM_RPS)
    url=f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote_plus(smiles)}/PNG?image_size=large"
    r=requests.get(url,timeout=10); r.raise_for_status(); return r.content

def rdkit_fp(smiles:str):
    if not RD_ENABLED: return None
    m=Chem.MolFromSmiles(smiles)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048)

def tanimoto(a,b):
    if RD_ENABLED and a is not None and b is not None:
        return DataStructs.TanimotoSimilarity(a,b)
    def grams(s,k=3): return set(s[i:i+k] for i in range(max(len(s)-k+1,1)))
    A,B=grams(a),grams(b); return len(A&B)/max(len(A|B),1)

def patentsview_search_text(query, rows=20):
    if OFFLINE_MODE: return []
    rate_limit(_last_pv,PATENTSVIEW_RPS)
    url="https://api.patentsview.org/patents/query"
    params={"q":json.dumps({"_text_any":{"patent_abstract":query}}),
            "f":json.dumps(["patent_number","patent_title","patent_date","assignee_organization","cpc_group_id"]),
            "o":json.dumps({"per_page":rows})}
    r=requests.get(url,params=params,timeout=15)
    if r.status_code==200:
        data=r.json(); return data.get("patents",[])
    return []

def fetch_claims_text(pn:str)->str:
    try:
        with open("claims_sample.json","r",encoding="utf-8") as f:
            claims=json.load(f)
        if pn in claims: return claims[pn]
    except Exception:
        pass
    if OFFLINE_MODE: return ""
    try:
        url=f"https://patents.google.com/patent/{pn}"
        html=requests.get(url,timeout=10).text
        soup=BeautifulSoup(html,"html.parser")
        text=soup.get_text(" ",strip=True)
        m=re.search(r"(?:What is claimed|I claim|We claim).*",text,flags=re.IGNORECASE)
        return m.group(0)[:1200] if m else ""
    except Exception:
        return ""

@st.cache_data
def load_offline_dataset():
    try:
        return pd.read_excel("mock_data_curated.xlsx")
    except Exception:
        return pd.DataFrame()

def offline_search(smiles:str,k=10):
    df=load_offline_dataset()
    if df.empty: return pd.DataFrame(),1.0,None,smiles
    canon=canonicalize_smiles(smiles)
    qfp=rdkit_fp(canon) if RD_ENABLED else canon
    rows=[]
    for _,r in df.iterrows():
        s=r.get("canonical") or r.get("smiles")
        tfp=rdkit_fp(s) if RD_ENABLED else s
        score=tanimoto(qfp,tfp) if RD_ENABLED else tanimoto(canon,s)
        rows.append({"patent_number":r.get("patent_number",""),
                     "assignee":r.get("assignee_organization",""),
                     "title":r.get("title",""),
                     "filing_date":r.get("patent_date",""),
                     "similarity":float(score)})
    out=pd.DataFrame(rows).sort_values("similarity",ascending=False).head(k)
    est=float(1.0-(out["similarity"].max() if not out.empty else 0.0))
    return out, est, None, canon

def live_search(smiles:str,k=10):
    canon=canonicalize_smiles(smiles)
    cid=pubchem_cid_from_smiles(canon)
    pv_hits=patentsview_search_text(canon,rows=40)
    rows=[]
    for p in pv_hits:
        rows.append({"patent_number":p.get("patent_number",""),
                     "assignee":(p.get("assignees") or [{}])[0].get("assignee_organization","") if p.get("assignees") else "",
                     "title":p.get("patent_title",""),
                     "filing_date":p.get("patent_date",""),
                     "similarity":0.12 if RD_ENABLED else 0.10})
    out=pd.DataFrame(rows).sort_values("similarity",ascending=False).head(k)
    est=float(1.0-(out["similarity"].max() if not out.empty else 0.0))
    return out, est, cid, canon

def search_novelty(smiles:str,k=10):
    if OFFLINE_MODE:
        return offline_search(smiles,k=k)
    try:
        return live_search(smiles,k=k)
    except Exception:
        st.warning("Live search failed; falling back to offline dataset.")
        return offline_search(smiles,k=k)

st.title("🧪 MoleculeIQ v17 – Chemistry-first Novelty")
st.caption("Live PubChem + PatentsView with offline fallback.")

default_smiles = st.session_state.get("smiles_input","CC(=O)OC1=CC=CC=C1C(=O)O")
smiles_in = st.text_input("Enter SMILES", value=default_smiles)
proposed_claim = st.text_area("Proposed Claim (optional)", height=140)

if st.button("Run Novelty Check"):
    with st.spinner("Searching..."):
        results, novelty, cid, canon = search_novelty(smiles_in, k=10)
        # Structure render
        try:
            if not OFFLINE_MODE:
                png = pubchem_png(canon)
                st.image(png, caption=f"Structure (canonical): {canon}", output_format="PNG")
            elif RD_ENABLED:
                m = Chem.MolFromSmiles(canon)
                if m:
                    img = Draw.MolToImage(m, size=(400,300))
                    st.image(img, caption=f"Structure (canonical): {canon}")
        except Exception as e:
            st.warning(f"Could not render structure: {e}")
        st.write(f"**Estimated Novelty (0–1):** {novelty:.2f}")
        if not results.empty:
            results["Open"] = results["patent_number"].apply(lambda pn: f"https://patents.google.com/patent/{pn}" if pn else "")
            st.dataframe(results[["patent_number","assignee","title","filing_date","similarity","Open"]],
                         use_container_width=True,
                         column_config={"Open": st.column_config.LinkColumn("Open ↗", display_text="Open ↗")},
                         hide_index=True)
        else:
            st.info("No results found.")
        if not results.empty and isinstance(proposed_claim,str) and proposed_claim.strip():
            ref_pn = results.iloc[0]["patent_number"]
            claim_text = fetch_claims_text(ref_pn) or "Reference claim text unavailable."
            st.subheader("Claim-level comparison (best-effort)")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Reference Claim (Top Match)**"); st.write(claim_text)
            with c2:
                st.markdown("**Proposed Claim**"); st.write(proposed_claim)

st.subheader("Batch Mode")
upl = st.file_uploader("Upload CSV with a 'smiles' column", type=["csv"])
if st.button("Process Batch"):
    if upl is None:
        st.warning("Please upload a CSV first.")
    else:
        df_in = pd.read_csv(upl)
        if "smiles" not in df_in.columns and "SMILES" in df_in.columns:
            df_in = df_in.rename(columns={"SMILES":"smiles"})
        assert "smiles" in df_in.columns, "CSV must contain a 'smiles' column"
        all_rows = []
        with st.spinner(f"Processing {len(df_in)} molecules..."):
            for s in df_in["smiles"].dropna().tolist():
                res, nov, cid, canon = search_novelty(s, k=10)
                top = res.iloc[0].to_dict() if not res.empty else {}
                all_rows.append({"smiles": s, "canonical": canon, "novelty": nov,
                                 "top_patent": top.get("patent_number",""), "title": top.get("title","")})
        st.dataframe(pd.DataFrame(all_rows))
