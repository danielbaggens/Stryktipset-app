import itertools, io, math, re
import numpy as np, openpyxl, pandas as pd, requests, streamlit as st

st.set_page_config(page_title="Stryktipset Pro", page_icon="⚽", layout="centered")
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_RADER_PER_FIL = 50000
API_URL = "https://api.spela." + "svenskaspel.se/draw/1/stryktipset/draws"

def tolka_marginal(varde):
    if varde is None or pd.isna(varde) or str(varde).strip() == "": return 0.0
    match = re.search(r"(\d+(?:[.,]\d+)?)", str(varde))
    return float(match.group(1).replace(",", ".")) if match else 0.0

@st.cache_data
def hamta_grundmatris():
    return np.indices((3,)*13, dtype=np.int8).reshape(13, -1).T

def hamta_live_kupong():
    resp = requests.get(API_URL, headers=HEADERS, timeout=10)
    draw = resp.json()["draws"][0]
    omgang = draw.get("drawNumber", "Okänd")
    matcher = []
    streck_matris = np.zeros((13, 3), dtype=np.float64)
    for idx, ev in enumerate(draw["drawEvents"][:13]):
        sf = ev.get("svenskaFolket", {})
        p1, px, p2 = float(sf.get("one", 33)), float(sf.get("x", 33)), float(sf.get("two", 34))
        streck_matris[idx] = [p1, px, p2]
        matcher.append({"Match": idx + 1, "Lag": ev.get("eventDescription", ""), "1 (%)": p1, "X (%)": px, "2 (%)": p2})
    return omgang, matcher, streck_matris

def kor_reducering_fran_df(df, streck_matris, extra_marginal=0.0):
    rader = hamta_grundmatris()
    rad_streck = streck_matris[np.arange(13), rader]
    aktiva_df = df[df["Aktiv"].astype(str).str.strip().str.upper() == "JA"]
    for _, row in aktiva_df.iterrows():
        if rader.shape[0] == 0: break
        typ = str(row["Typ"]).strip().upper()
        v1, v2, p_min, p_max = row["Varde1"], row["Varde2"], row["ProcentMin"], row["ProcentMax"]
        marg = tolka_marginal(row["Marginal"]) + extra_marginal
        if typ == "ANTAL_1": s = np.sum(rader==0, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "ANTAL_X": s = np.sum(rader==1, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "ANTAL_2": s = np.sum(rader==2, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "TECKENBYTEN": b = np.sum(rader[:,:-1]!=rader[:,1:], axis=1); ok = (b >= float(v1)) & (b <= float(v2))
        elif typ == "ANTAL_PAR": p = np.sum(rader[:,:-1]==rader[:,1:], axis=1); ok = (p >= float(v1)) & (p <= float(v2))
        elif typ == "STRECKSUMMA": ss = np.sum(rad_streck, axis=1); ok = (ss >= float(v1)) & (ss <= float(v2))
        elif typ == "TOPP_STRECKADE":
            tr = np.partition(streck_matris.flatten(), -int(p_min))[-int(p_min)]
            t = np.sum(rad_streck >= tr, axis=1); ok = (t >= float(v1)) & (t <= float(v2))
        elif typ == "BOTTEN_STRECKADE":
            tr = np.partition(streck_matris.flatten(), int(p_min)-1)[int(p_min)-1]
            t = np.sum(rad_streck <= tr, axis=1); ok = (t >= float(v1)) & (t <= float(v2))
        elif typ in ("INTERVALL_1", "INTERVALL_X", "INTERVALL_2", "INTERVALL_ALLA"):
            in_int = (rad_streck >= max(0.0, float(p_min)-marg)) & (rad_streck <= min(100.0, float(p_max)+marg))
            if typ == "INTERVALL_1": t = np.sum((rader==0) & in_int, axis=1)
            elif typ == "INTERVALL_X": t = np.sum((rader==1) & in_int, axis=1)
            elif typ == "INTERVALL_2": t = np.sum((rader==2) & in_int, axis=1)
            else: t = np.sum(in_int, axis=1)
            ok = (t >= float(v1)) & (t <= float(v2))
        else: continue
        rader, rad_streck = rader[ok], rad_streck[ok]
    return rader, len(aktiva_df)
st.title("⚽ Stryktipset Mobilmotor")

uppladdad_fil = st.file_uploader("Ladda upp din villkor.xlsx", type=["xlsx"])
if uppladdad_fil is not None:
    if "senaste_filnamn" not in st.session_state or st.session_state["senaste_filnamn"] != uppladdad_fil.name:
        df_in = pd.read_excel(uppladdad_fil, sheet_name="Villkor")
        for col in ["Aktiv", "Typ", "Varde1", "Varde2", "ProcentMin", "ProcentMax", "Marginal", "Källa / Kommentar"]:
            if col not in df_in.columns: df_in[col] = ""
        st.session_state["villkor_df"] = df_in[["Aktiv", "Typ", "Varde1", "Varde2", "ProcentMin", "ProcentMax", "Marginal", "Källa / Kommentar"]]
        st.session_state["senaste_filnamn"] = uppladdad_fil.name
elif "villkor_df" not in st.session_state:
    st.session_state["villkor_df"] = pd.DataFrame([
        {"Aktiv": "JA", "Typ": "TOPP_STRECKADE", "Varde1": 1, "Varde2": 4, "ProcentMin": 5, "ProcentMax": None, "Marginal": "", "Källa / Kommentar": "Av 5 mest spelade"},
        {"Aktiv": "JA", "Typ": "STRECKSUMMA", "Varde1": 420, "Varde2": 630, "ProcentMin": None, "ProcentMax": None, "Marginal": "", "Källa / Kommentar": "Total strecksumma"},
    ])

st.subheader("🎛️ Snabbjustering & Redigerare")
extra_marginal = st.slider("Extra marginal på alla intervall (± procentenheter)", -3.0, 5.0, 0.0, 0.5)

with st.expander("✏️ Öppna villkorsredigeraren", expanded=False):
    filter_text = st.text_input("Sök villkor (t.ex. INTERVALL_1 eller 45)", "")
    df_att_visa = st.session_state["villkor_df"]
    if filter_text:
        mask = df_att_visa.apply(lambda r: r.astype(str).str.contains(filter_text, case=False).any(), axis=1)
        redigerad_del = st.data_editor(df_att_visa[mask], num_rows="dynamic", use_container_width=True)
        st.session_state["villkor_df"].update(redigerad_del)
    else:
        st.session_state["villkor_df"] = st.data_editor(df_att_visa, num_rows="dynamic", use_container_width=True)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        st.session_state["villkor_df"].to_excel(writer, sheet_name="Villkor", index=False)
    st.download_button("💾 Spara uppdaterad villkor.xlsx", excel_buffer.getvalue(), "villkor_uppdaterad.xlsx", use_container_width=True)

st.divider()
if st.button("🔄 Hämta live-procent & Beräkna nu", type="primary", use_container_width=True):
    with st.spinner("Hämtar streckprocent & räknar igenom 1 594 323 rader..."):
        omgang, matcher, streck_matris = hamta_live_kupong()
        godkanda, antal_aktiva = kor_reducering_fran_df(st.session_state["villkor_df"], streck_matris, extra_marginal)
        kostnad = godkanda.shape[0]

        st.success(f"Omgång {omgang} beräknad med {antal_aktiva} aktiva villkor!")
        c1, c2 = st.columns(2)
        c1.metric("Systemkostnad", f"{kostnad:,} kr".replace(",", " "))
        c2.metric("Antal rader", f"{kostnad:,} st".replace(",", " "))

        if kostnad > 0:
            tecken_map = np.array(["1", "X", "2"])
            antal_filer = math.ceil(kostnad / MAX_RADER_PER_FIL)
            for del_nr in range(antal_filer):
                del_rader = tecken_map[godkanda[del_nr * MAX_RADER_PER_FIL : min((del_nr + 1) * MAX_RADER_PER_FIL, kostnad)]]
                filnamn = "stryktipset_system.txt" if antal_filer == 1 else f"stryktipset_del{del_nr + 1}.txt"
                fil_innehall = "Stryktipset\n" + "\n".join("E," + ",".join(r) for r in del_rader)
                st.download_button(f"📥 1. Ladda ner {filnamn} ({len(del_rader):,} rader)".replace(",", " "), fil_innehall, filnamn, "text/plain", use_container_width=True, key=f"dl_{del_nr}")

            st.link_button("🚀 2. Öppna Spela Tillsammans", "https://svenskaspel.se", use_container_width=True)
            st.link_button("⚽ Alt: Öppna vanliga Stryktipset", "https://svenskaspel.se", use_container_width=True)
        else:
            st.warning("0 rader klarade villkoren! Dra reglaget 'Extra marginal' åt höger och beräkna igen.")

        with st.expander("Visa veckans live-procent"):
            st.dataframe(matcher, use_container_width=True)
