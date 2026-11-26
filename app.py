import io, math, re
import numpy as np, openpyxl, pandas as pd, requests, streamlit as st

st.set_page_config(page_title="Stryktipset Pro", page_icon="⚽", layout="centered")
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_RADER_PER_FIL = 2200

SPEL_INFO = {
    "⚽ Stryktipset (13 matcher)": {
        "api": "https://api.spela." + "svenskaspel.se/draw/1/stryktipset/draws",
        "matcher": 13,
        "flik": "Villkor",
        "rubrik": "Stryktipset",
        "url_tills": "https://svenskaspel.se",
        "url_egen": "https://svenskaspel.se",
    },
    "🌍 Europatipset (13 matcher)": {
        "api": "https://api.spela." + "svenskaspel.se/draw/1/europatipset/draws",
        "matcher": 13,
        "flik": "Villkor_Europa",
        "rubrik": "Europatipset",
        "url_tills": "https://svenskaspel.se",
        "url_egen": "https://svenskaspel.se",
    },
    "🎯 Topptipset (8 matcher)": {
        "api": "https://api.spela." + "svenskaspel.se/draw/1/topptipsetfamily/draws",
        "matcher": 8,
        "flik": "Villkor_Topptipset",
        "rubrik": "Topptipset",
        "url_tills": "https://svenskaspel.se",
        "url_egen": "https://svenskaspel.se",
    },
}

def tolka_marginal(varde):
    if varde is None or pd.isna(varde) or str(varde).strip() == "": return 0.0
    match = re.search(r"(\d+(?:[.,]\d+)?)", str(varde))
    return float(match.group(1).replace(",", ".")) if match else 0.0

@st.cache_data
def hamta_grundmatris(n_matcher):
    return np.indices((3,)*n_matcher, dtype=np.int8).reshape(n_matcher, -1).T

def hamta_live_kupong(api_url, n_matcher):
    resp = requests.get(api_url, headers=HEADERS, timeout=10)
    draws = resp.json().get("draws", [])
    draw = draws.pop(0)
    omgang = draw.get("drawNumber", "Okänd")
    namn = draw.get("drawComment", "")
    matcher = []
    streck_matris = np.zeros((n_matcher, 3), dtype=np.float64)
    for idx, ev in enumerate(draw["drawEvents"][:n_matcher]):
        sf = ev.get("svenskaFolket", {})
        p1, px, p2 = float(sf.get("one", 33)), float(sf.get("x", 33)), float(sf.get("two", 34))
        streck_matris[idx] = [p1, px, p2]
        matcher.append({"Match": idx + 1, "Lag": ev.get("eventDescription", ""), "1 (%)": p1, "X (%)": px, "2 (%)": p2})
    return f"{omgang} ({namn})" if namn else str(omgang), matcher, streck_matris

def kor_reducering_fran_df(df, streck_matris, n_matcher, extra_marginal=0.0):
    rader = hamta_grundmatris(n_matcher)
    rad_streck = streck_matris[np.arange(n_matcher), rader]
    rank_matris = np.argsort(np.argsort(-streck_matris, axis=1), axis=1)
    rad_rank = rank_matris[np.arange(n_matcher), rader]
    aktiva_df = df[df["Aktiv"].astype(str).str.strip().str.upper() == "JA"].copy()
    aktiva_df["_prio"] = aktiva_df["Typ"].astype(str).str.strip().str.upper().apply(lambda t: 1 if t.startswith("INTERVALL") else 0)
    aktiva_df = aktiva_df.sort_values("_prio")

    for _, row in aktiva_df.iterrows():
        if rader.shape.__getitem__(0) == 0: break
        typ = str(row["Typ"]).strip().upper()
        v1, v2, p_min, p_max = row["Varde1"], row["Varde2"], row["ProcentMin"], row["ProcentMax"]
        marg = tolka_marginal(row["Marginal"]) + extra_marginal
        if typ == "ANTAL_1": s = np.sum(rader==0, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "ANTAL_X": s = np.sum(rader==1, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "ANTAL_2": s = np.sum(rader==2, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "ANTAL_FAVORITER": s = np.sum(rad_rank==0, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "ANTAL_ANDRAHAND": s = np.sum(rad_rank==1, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "ANTAL_SKRALLAR": s = np.sum(rad_rank==2, axis=1); ok = (s >= float(v1)) & (s <= float(v2))
        elif typ == "POANGSUMMA": p = np.sum(rad_rank + 1, axis=1); ok = (p >= float(v1)) & (p <= float(v2))
        elif typ == "RAD_ODDS":
            odds = 1.0 / np.prod(np.maximum(rad_streck, 1.0) / 100.0, axis=1)
            ok = (odds >= float(v1)) & (odds <= float(v2))
        elif typ == "FALLANDE_FAVORITER":
            grans = float(p_min) if pd.notna(p_min) and str(p_min).strip() != "" else 65.0
            har_stor_fav = (np.max(streck_matris, axis=1) >= grans)
            foll = np.sum((rad_rank > 0) & har_stor_fav, axis=1)
            ok = (foll >= float(v1)) & (foll <= float(v2))
        elif typ == "SKRALL_SUMMA":
            ss = np.sum(np.where(rad_rank > 0, rad_streck, 0.0), axis=1)
            ok = (ss >= float(v1)) & (ss <= float(v2))
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
        rader, rad_streck, rad_rank = rader[ok], rad_streck[ok], rad_rank[ok]
    return rader, len(aktiva_df)
st.title("⚽ Svenska Spel Mobilmotor")

valt_spel = st.radio("Välj kupong:", list(SPEL_INFO.keys()), horizontal=True)
cfg = SPEL_INFO.get(valt_spel)
flik_namn = cfg.get("flik")
n_matcher = cfg.get("matcher")

uppladdad_fil = st.file_uploader("Ladda upp din villkor.xlsx", type=["xlsx"])
nyckel_state = f"df_{flik_namn}"

if uppladdad_fil is not None:
    fil_id = f"{uppladdad_fil.name}_{uppladdad_fil.size}_{flik_namn}"
    if st.session_state.get("senaste_id") != fil_id:
        xl = pd.ExcelFile(uppladdad_fil)
        mal_flik = flik_namn if flik_namn in xl.sheet_names else "Villkor"
        df_in = xl.parse(mal_flik)
        for col in ["Aktiv", "Typ", "Varde1", "Varde2", "ProcentMin", "ProcentMax", "Marginal", "Källa / Kommentar"]:
            if col not in df_in.columns: df_in[col] = ""
        st.session_state[nyckel_state] = df_in[["Aktiv", "Typ", "Varde1", "Varde2", "ProcentMin", "ProcentMax", "Marginal", "Källa / Kommentar"]]
        st.session_state["senaste_id"] = fil_id

if nyckel_state not in st.session_state:
    if n_matcher == 8:
        st.session_state[nyckel_state] = pd.DataFrame([
            {"Aktiv": "JA", "Typ": "STRECKSUMMA", "Varde1": 275, "Varde2": 365, "ProcentMin": None, "ProcentMax": None, "Marginal": "", "Källa / Kommentar": "Topptipset strecksumma"},
            {"Aktiv": "JA", "Typ": "POANGSUMMA", "Varde1": 11, "Varde2": 18, "ProcentMin": None, "ProcentMax": None, "Marginal": "", "Källa / Kommentar": "Rankpoäng 11-18"},
        ])
    else:
        st.session_state[nyckel_state] = pd.DataFrame([
            {"Aktiv": "JA", "Typ": "STRECKSUMMA", "Varde1": 440, "Varde2": 570, "ProcentMin": None, "ProcentMax": None, "Marginal": "", "Källa / Kommentar": "Strecksumma 13 matcher"},
            {"Aktiv": "JA", "Typ": "POANGSUMMA", "Varde1": 19, "Varde2": 28, "ProcentMin": None, "ProcentMax": None, "Marginal": "", "Källa / Kommentar": "Rankpoäng 19-28"},
        ])

st.subheader(f"🎛️ Inställningar ({flik_namn})")
extra_marginal = st.slider("Extra marginal på alla intervall (± procentenheter)", -3.0, 5.0, 0.0, 0.5)

with st.expander("✏️ Öppna villkorsredigeraren", expanded=False):
    filter_text = st.text_input("Sök villkor", "")
    df_att_visa = st.session_state[nyckel_state]
    if filter_text:
        mask = df_att_visa.apply(lambda r: r.astype(str).str.contains(filter_text, case=False).any(), axis=1)
        redigerad_del = st.data_editor(df_att_visa[mask], num_rows="dynamic", use_container_width=True)
        st.session_state[nyckel_state].update(redigerad_del)
    else:
        st.session_state[nyckel_state] = st.data_editor(df_att_visa, num_rows="dynamic", use_container_width=True)

st.divider()
if st.button(f"🔄 Hämta {cfg.get('rubrik')} & Beräkna nu", type="primary", use_container_width=True):
    with st.spinner(f"Hämtar {cfg.get('rubrik')} & räknar igenom {3**n_matcher:,} rader...".replace(",", " ")):
        omgang, matcher, streck_matris = hamta_live_kupong(cfg.get("api"), n_matcher)
        aktiv_marg = extra_marginal
        godkanda, antal_aktiva = kor_reducering_fran_df(st.session_state[nyckel_state], streck_matris, n_matcher, aktiv_marg)
        while godkanda.shape.__getitem__(0) > 20000 and aktiv_marg > -3.0:
            aktiv_marg -= 0.5
            godkanda, antal_aktiva = kor_reducering_fran_df(st.session_state[nyckel_state], streck_matris, n_matcher, aktiv_marg)
        if godkanda.shape.__getitem__(0) > 20000:
            godkanda = godkanda[:20000]
        kostnad = godkanda.shape.__getitem__(0)

        st.success(f"{cfg.get('rubrik')} omgång {omgang} beräknad med {antal_aktiva} aktiva villkor (Marginal: {aktiv_marg})!")
        c1, c2 = st.columns(2)
        c1.metric("Systemkostnad", f"{kostnad:,} kr".replace(",", " "))
        c2.metric("Antal rader", f"{kostnad:,} st".replace(",", " "))

        if kostnad > 2200:
            st.warning(f"⚠️ {kostnad:,} rader överstiger 2 200 rader för en fil. Systemet delas upp med 2 200, 2 199 osv. rader per fil.".replace(",", " "))

        if kostnad > 0:
            tecken_map = np.array(["1", "X", "2"])
            prefix = cfg.get("rubrik").lower()
            start_idx = 0
            del_nr = 1
            while start_idx < kostnad:
                max_for_denna_del = max(1, MAX_RADER_PER_FIL - (del_nr - 1))
                slut_idx = min(start_idx + max_for_denna_del, kostnad)
                del_rader = tecken_map[godkanda[start_idx:slut_idx]]
                antal_i_del = len(del_rader)
                filnamn = f"{prefix}_system.txt" if (del_nr == 1 and slut_idx == kostnad) else f"{prefix}_del{del_nr}_{antal_i_del}rader.txt"
                fil_innehall = f"{cfg.get('rubrik')}\n" + "\n".join("E," + ",".join(r) for r in del_rader)
                st.download_button(f"📥 Del {del_nr}: Ladda ner {filnamn} ({antal_i_del:,} rader)".replace(",", " "), fil_innehall, filnamn, "text/plain", use_container_width=True, key=f"dl_{del_nr}")
                start_idx = slut_idx
                del_nr += 1

            st.link_button("🚀 2. Öppna Spela Tillsammans", cfg.get("url_tills"), use_container_width=True)
            st.link_button(f"⚽ Alt: Öppna {cfg.get('rubrik')}", cfg.get("url_egen"), use_container_width=True)
        else:
            st.warning("0 rader klarade villkoren! Dra reglaget 'Extra marginal' åt höger och beräkna igen.")

        with st.expander("Visa veckans live-procent"):
            st.dataframe(matcher, use_container_width=True)
