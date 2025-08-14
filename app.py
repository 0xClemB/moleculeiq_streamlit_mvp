
import streamlit as st
import pandas as pd
import numpy as np
import hashlib, re, json, base64, os, hmac
from datetime import datetime
from io import BytesIO

# === PBKDF2-HMAC password hashing (stdlib; no external deps) ===
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


# =================== Simple Login (username/password via bcrypt) ===================
def load_users():
    try:
        with open("users.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

USERS = load_users()

def check_password(username: str, password: str) -> bool:
    user = USERS.get(username)
    if not user:
        return False
    try:
        return verify_password(password, user["password_hash"]), user["password_hash"].encode("utf-8"))
    except Exception:
        return False

def login_gate():
    st.title("🔐 MoleculeIQ Login")
    st.caption("For testing only — use one of the provided demo accounts.")
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="e.g., scientist")
        password = st.text_input("Password", type="password", placeholder="e.g., Chem!2025Test")
        submitted = st.form_submit_button("Login")
        if submitted:
            if check_password(username, password):
                st.session_state["user"] = USERS[username]["name"]
                st.session_state["role"] = USERS[username]["role"]
                st.success(f"Welcome {USERS[username]['name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# If not logged in, show login form and stop app
if "user" not in st.session_state:
    login_gate()
    st.stop()

# =================== App continues below if authenticated ===================

# Optional RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RD_ENABLED = True
except Exception:
    RD_ENABLED = False

st.set_page_config(page_title="MoleculeIQ – Patent Novelty MVP", page_icon="🧪", layout="wide")

# Sidebar user info + logout
with st.sidebar:
    st.markdown("### 👤 User")
    st.write(st.session_state.get("user", "Unknown"))
    st.caption(st.session_state.get("role", ""))
    if st.button("Logout"):
        for k in ["user","role"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

# === Sidebar: Examples dropdown ===
example_smiles = {
    "Methane": "C",
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Cholesterol": "C[C@H](CCC(=O)O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Penicillin G": "CC1(C)S[C@H]2[C@H](NC1=O)C(=O)N2C(=O)C(C1=CC=CC=C1)O",
    "Morphine": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@H](O)C=C[C@H]3[C@@H]1C5",
    "Serotonin": "C1=CC2=C(C=C1O)C(=CN2)CCN"
}
with st.sidebar:
    st.markdown("### Examples")
    example_choice = st.selectbox("Select an example", list(example_smiles.keys()), index=None, placeholder="Choose...")
    if example_choice:
        st.session_state["smiles_input"] = example_smiles[example_choice]



@st.cache_data
def load_data():
    return pd.read_csv("mock_data_curated.csv")

@st.cache_data
def load_claims_sample():
    try:
        with open("claims_sample.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

df = load_data()
claims_local = load_claims_sample()

def smiles_to_vec(smiles: str, dim: int = 128) -> np.ndarray:
    h = hashlib.sha256(smiles.encode()).digest()
    raw = (h * (dim // len(h) + 1))[:dim]
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr

@st.cache_data
def index_embeddings(smiles_list, dim=128):
    vecs = np.vstack([smiles_to_vec(s, dim) for s in smiles_list])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms

emb_index = index_embeddings(df["smiles"].tolist())

def search(smiles_query: str, k: int = 10):
    q = smiles_to_vec(smiles_query)
    q = q / (np.linalg.norm(q) + 1e-8)
    sims = emb_index @ q
    idx = np.argsort(-sims)[:k]
    results = df.iloc[idx].copy()
    results["similarity"] = sims[idx]
    novelty_score = float(1.0 - results["similarity"].max())
    return results, novelty_score

def risk_label(novelty: float):
    if novelty >= 0.6:
        return "🟢 Low"
    elif novelty >= 0.35:
        return "🟠 Moderate"
    else:
        return "🔴 High"

def fetch_structure_image(query_smiles: str, size=(600, 400)):
    """Return bytes of a PNG structure image. RDKit if available; else download from PubChem."""
    if RD_ENABLED:
        try:
            mol = Chem.MolFromSmiles(query_smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            img = Draw.MolToImage(mol, size=size)
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return buf.read()
        except Exception:
            pass
    # Fallback: PubChem PNG download
    import requests
    from urllib.parse import quote
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(query_smiles)}/PNG?image_size=large"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.content

# ---------- Patent links ----------
def google_patent_direct_url(pn: str) -> str:
    return f"https://patents.google.com/patent/{pn.replace(' ', '')}"

def google_patent_search_url(pn: str) -> str:
    from urllib.parse import quote_plus
    return f"https://patents.google.com/?q={quote_plus(pn)}"

@st.cache_data(show_spinner=False)
def resolve_patent_link(pn: str) -> str:
    import requests
    try:
        url = google_patent_direct_url(pn)
        resp = requests.head(url, timeout=6, allow_redirects=True)
        if resp.status_code >= 400 or resp.status_code == 405:
            resp = requests.get(url, timeout=6, allow_redirects=True)
        if 200 <= resp.status_code < 400:
            return url
    except Exception:
        pass
    return google_patent_search_url(pn)

# ---------- Claim fetch ----------
@st.cache_data(show_spinner=False)
def fetch_claim_text(pn: str) -> str:
    if pn in claims_local:
        return claims_local[pn]
    # best-effort minimal scrape
    import requests, bs4
    try:
        url = google_patent_direct_url(pn)
        html = requests.get(url, timeout=10).text
        soup = bs4.BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        m = re.search(r"(What is claimed.*?)(?:\\.[\\s]*\\d+\\.|Claim 1|1\\.)", text, flags=re.IGNORECASE)
        if m:
            return m.group(1)[:1000]
    except Exception:
        pass
    return ""

# ---------- PDF builders ----------
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image as RLImage, PageBreak
from reportlab.platypus.flowables import AnchorFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm

def header_footer(canvas, doc):
    width, height = doc.pagesize
    canvas.setFillColorRGB(0.06, 0.09, 0.16)
    canvas.rect(0, height-22*mm, width, 22*mm, fill=1, stroke=0)
    canvas.setFillColorRGB(1, 1, 1)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(20*mm, height-12*mm, "MoleculeIQ – Patent Novelty Report (MOCK)")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColorRGB(0.2, 0.2, 0.25)
    canvas.drawString(20*mm, 12*mm, "© MoleculeIQ – For validation only. Not legal advice.")
    page_num = canvas.getPageNumber()
    canvas.drawRightString(width-20*mm, 12*mm, f"Page {page_num}")

def make_sim_table(df_res: pd.DataFrame):
    header = ["Patent", "Assignee", "Title", "Filing Date", "Similarity", "Open"]
    data = [header]
    for _, row in df_res.iterrows():
        url = resolve_patent_link(row["patent_number"])
        patent_link = Paragraph(f'<link href="{url}">{row["patent_number"]}</link>', ParagraphStyle("link", textColor=colors.HexColor("#1D4ED8")))
        title_para = Paragraph(row["title"], ParagraphStyle("wrap", fontSize=9, leading=11))
        open_link = Paragraph(f'<link href="{url}">Open ↗</link>', ParagraphStyle("link2", textColor=colors.HexColor("#1D4ED8")))
        data.append([patent_link, row["assignee"], title_para, row["filing_date"], f"{row['similarity']:.2f}", open_link])
    tbl = Table(data, colWidths=[30*mm, 30*mm, 82*mm, 28*mm, 22*mm, 18*mm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor("#F8FAFC")]),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return tbl

def build_pdf_enterprise(query_smiles, results_df, novelty, png_bytes, proposed_claim_text=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=28*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14, spaceBefore=8, spaceAfter=4)
    wrap = ParagraphStyle("wrap", parent=styles["Normal"], fontSize=9, leading=11)

    story = []
    story.append(AnchorFlowable("toc"))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", sub))
    story.append(Paragraph(f"Query SMILES: <b>{query_smiles}</b>", styles["Normal"]))
    story.append(Paragraph(f"Estimated Novelty (0–1): <b>{novelty:.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Patentability Risk: <b>{risk_label(novelty)}</b>", styles["Normal"]))
    story.append(Spacer(1, 6*mm))
    story.append(RLImage(BytesIO(png_bytes), width=90*mm, height=60*mm))
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("Top Similar Patents", h2))
    story.append(make_sim_table(results_df))

    if proposed_claim_text and isinstance(proposed_claim_text, str) and proposed_claim_text.strip():
        ref_pn = str(results_df.iloc[0]["patent_number"])
        ref_claim = fetch_claim_text(ref_pn) or "Reference claim text unavailable."
        ref_claim_short = (ref_claim[:1200] + "…") if len(ref_claim) > 1200 else ref_claim

        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("Claim-level comparison (vs. nearest prior art)", h2))
        story.append(Paragraph(f"Reference patent: <link href='{google_patent_direct_url(ref_pn)}'>{ref_pn} ↗</link>", styles["Normal"]))

        claim_table = Table([
            ["Reference Claim", "Proposed Claim"],
            [Paragraph(ref_claim_short.replace("\n"," "), wrap),
             Paragraph(proposed_claim_text.replace("\n"," "), wrap)]
        ], colWidths=[90*mm, 90*mm])
        claim_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10),
        ]))
        story.append(claim_table)

        story.append(PageBreak())
        story.append(Paragraph("Diff Legend", styles["Heading1"]))
        story.append(Paragraph("This legend explains how to interpret differences in the comparison.", styles["Normal"]))
        legend = Table([
            ["Style", "Meaning"],
            ["🟩 Green highlight", "New content or additions in the proposed claim."],
            ["🔴 Red strikethrough", "Content present in the reference claim but removed in the proposed claim."],
            ["🟧 Amber highlight", "Content that has been modified or replaced compared to the reference claim."]
        ], colWidths=[60*mm, 120*mm])
        legend.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10),
        ]))
        story.append(legend)

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph('<link href="#toc">Back to TOC ↩</link>', ParagraphStyle("back", textColor=colors.HexColor("#1D4ED8"))))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    buffer = BytesIO()
    buffer.write(doc.canv.getpdfdata() if hasattr(doc, 'canv') else b"")
    # The above may be empty depending on reportlab version; instead return the file-like buffer from build.
    # We'll re-build to get bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=28*mm, bottomMargin=20*mm)
    story2 = []
    story2.extend([])  # no-op, but ensures variable exists
    # Rebuild actual content
    story2 = story
    doc.build(story2, onFirstPage=header_footer, onLaterPages=header_footer)
    buffer.seek(0)
    return buffer

from reportlab.platypus import PageBreak
def build_consolidated_pdf(batch_rows: list):
    class TOCDoc(SimpleDocTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sections = []
        def afterFlowable(self, flowable):
            if hasattr(flowable, "bookmarkName"):
                self.sections.append((getattr(flowable, "toc_text", flowable.bookmarkName),
                                      flowable.bookmarkName, self.canv.getPageNumber()))

    def make_story(include_toc=False, toc_data=None):
        styles = getSampleStyleSheet()
        sub_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
        h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=8, spaceAfter=4)
        hsec = ParagraphStyle("Hsec", parent=styles["Heading2"], fontSize=13, spaceBefore=6, spaceAfter=6)
        link_style = ParagraphStyle("Link", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#1D4ED8"))
        wrap_style = ParagraphStyle("Wrap", parent=styles["Normal"], fontSize=9, leading=11)

        story = []
        if include_toc and toc_data is not None:
            story.append(AnchorFlowable("toc"))
            story.append(Paragraph("Table of Contents", styles["Heading1"]))
            data = [["#", "Molecule (SMILES)", "Page"]]
            for i, (title, key, page) in enumerate(toc_data, start=1):
                link = Paragraph(f'<link href="#{key}">{title}</link>', link_style)
                data.append([str(i), link, str(page)])
            toc_table = Table(data, colWidths=[12*mm, 140*mm, 20*mm])
            toc_table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("ALIGN", (0,0), (-1,0), "CENTER"),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("FONTSIZE", (0,0), (-1,0), 11),
                ("FONTSIZE", (0,1), (-1,-1), 10),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(toc_table)
            story.append(PageBreak())

        for idx, item in enumerate(batch_rows):
            query_smiles = item["smiles"]
            novelty = item["novelty"]
            results_df = item["results"]
            png_bytes = item["png_bytes"]

            key = f"sec_{idx+1}"
            story.append(AnchorFlowable(key))
            sec_title = Paragraph(f"Molecule {idx+1}: <b>{query_smiles}</b>", hsec)
            sec_title.bookmarkName = key
            sec_title.toc_text = query_smiles
            story.append(sec_title)
            story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", sub_style))
            story.append(Spacer(1, 3*mm))
            story.append(Paragraph(f"Estimated Novelty (0–1): <b>{novelty:.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Patentability Risk: <b>{risk_label(novelty)}</b>", styles["Normal"]))
            story.append(Spacer(1, 6*mm))
            story.append(RLImage(BytesIO(png_bytes), width=90*mm, height=60*mm))
            story.append(Spacer(1, 6*mm))

            story.append(Paragraph("Top Similar Patents", h2))
            story.append(make_sim_table(results_df))

            story.append(Spacer(1, 2*mm))
            story.append(Paragraph('<link href="#toc">Back to TOC ↩</link>', ParagraphStyle("back", textColor=colors.HexColor("#1D4ED8"))))

            if idx < len(batch_rows) - 1:
                story.append(PageBreak())

        return story

    buf1 = BytesIO()
    doc1 = TOCDoc(buf1, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=28*mm, bottomMargin=20*mm)
    story1 = make_story(False)
    doc1.build(story1, onFirstPage=header_footer, onLaterPages=header_footer)
    sections = doc1.sections

    buffer = BytesIO()
    doc2 = TOCDoc(buffer, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=28*mm, bottomMargin=20*mm)
    story2 = make_story(True, sections)
    doc2.build(story2, onFirstPage=header_footer, onLaterPages=header_footer)
    buffer.seek(0)
    return buffer

# ---- UI ----
st.title("🧪 MoleculeIQ – Patent Novelty (Demo)")
st.write("Paste a **SMILES** string and (optionally) a proposed claim to get a mock novelty report.")

prefill = "CC(=O)OC1=CC=CC=C1C(=O)O"
default_smiles = st.session_state.get("smiles_input", prefill)
query = st.text_input("Enter SMILES", value=default_smiles)
proposed_claim = st.text_area("Proposed Claim (optional)", height=140, placeholder="e.g., 1. A pharmaceutical composition comprising ...")
go = st.button("Run Novelty Check")

if go and query.strip():
    with st.spinner("Computing similarity and generating report..."):
        results, novelty = search(query.strip(), k=10)

    try:
        png_bytes = fetch_structure_image(query.strip())
        st.image(png_bytes, caption="Query Structure", output_format="PNG")
    except Exception as e:
        st.warning(f"Could not render structure image: {e}")
        png_bytes = None

    st.subheader("Top Similar Patents (Mock)")
    results = results.copy()
    results["patent_url"] = results["patent_number"].map(resolve_patent_link)
    results["Open"] = results["patent_url"]
    show = results[["patent_number","assignee","title","filing_date","similarity","Open"]].copy()
    show["similarity"] = show["similarity"].map(lambda x: f"{x:.2f}")
    st.dataframe(
        show,
        use_container_width=True,
        column_config={
            "patent_number": "Patent",
            "assignee": "Assignee",
            "title": "Title",
            "filing_date": "Filing Date",
            "similarity": "Similarity",
            "Open": st.column_config.LinkColumn("Open ↗", display_text="Open ↗")
        },
        hide_index=True
    )

    st.subheader("Export")
    if png_bytes is None:
        try: png_bytes = fetch_structure_image(query.strip())
        except Exception: png_bytes = b''
    try:
        pdf_buf = build_pdf_enterprise(query.strip(), results, novelty, png_bytes, proposed_claim_text=proposed_claim)
        st.download_button("Download PDF Report", pdf_buf.getvalue(), "moleculeiq_novelty_report.pdf", "application/pdf")
    except Exception as e:
        st.warning(f"PDF export unavailable: {e}")

# Batch mode
st.subheader("Batch Mode (CSV → Consolidated PDF with TOC + Back-to-TOC links)")
uploaded = st.file_uploader("Upload CSV with a 'smiles' column", type=["csv"])
if st.button("Process Batch"):
    if uploaded is None:
        st.warning("Please upload a CSV first.")
    else:
        try:
            df_in = pd.read_csv(uploaded)
            if "smiles" not in df_in.columns and "SMILES" in df_in.columns:
                df_in = df_in.rename(columns={"SMILES": "smiles"})
            assert "smiles" in df_in.columns, "CSV must contain a 'smiles' or 'SMILES' column"
            df_in = df_in.dropna(subset=["smiles"]).head(30)
            batch_rows = []
            with st.spinner(f"Processing {len(df_in)} molecules..."):
                for s in df_in["smiles"]:
                    s = str(s).strip()
                    res, nov = search(s, k=10)
                    try:
                        png = fetch_structure_image(s)
                    except Exception:
                        png = None
                    if png is None:
                        png = b'\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f\\x15\\xc4\\x89\\x00\\x00\\x00\\x0cIDATx\\x9cc````\\x00\\x00\\x00\\x05\\x00\\x01\\r\\n-\\xb4\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82'
                    batch_rows.append({"smiles": s, "results": res, "novelty": nov, "png_bytes": png})
            pdf_all = build_consolidated_pdf(batch_rows)
            st.download_button("Download Consolidated PDF (with TOC)", pdf_all.getvalue(), "moleculeiq_batch_report.pdf", "application/pdf")
        except Exception as e:
            st.error(f"Batch processing failed: {e}")
