import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib

# Konfiguracja strony
st.set_page_config(page_title="Predykcja niewydolności serca", page_icon="❤️")


# Funkcja do ładowania modelu
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        with open('feature_info.json', 'r') as f:
            feature_info = json.load(f)
        return model, scaler, feature_info
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {e}")
        return None, None, None


# Inicjalizacja Gemini API (opcjonalne)
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# Tytuł aplikacji
st.title("❤️ System predykcji niewydolności serca")

# Sidebar dla Gemini
api_key = None
gemini_model = None

if GEMINI_AVAILABLE:
    with st.sidebar:
        st.header("Ustawienia LLM")
        api_key = st.text_input("Klucz API Gemini", type="password")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                st.success("Połączono z Gemini API")
            except Exception as e:
                st.error(f"Błąd Gemini: {e}")
                gemini_model = None

# Ładowanie modelu
model, scaler, feature_info = load_model()

if model is None:
    st.error(
        "Nie można załadować modelu. Upewnij się, że pliki model.joblib, scaler.joblib i feature_info.json istnieją.")
    st.stop()

# Formularz danych
st.header("Dane pacjenta")
col1, col2 = st.columns(2)

# Dane pacjenta
with col1:
    age = st.number_input("Wiek", min_value=int(feature_info['ranges']['age'][0]),
                          max_value=int(feature_info['ranges']['age'][1]), value=65)
    sex = st.selectbox("Płeć", options=[0, 1], format_func=lambda x: "Kobieta" if x == 0 else "Mężczyzna")
    anaemia = st.selectbox("Anemia", options=[0, 1], format_func=lambda x: "Nie" if x == 0 else "Tak")
    diabetes = st.selectbox("Cukrzyca", options=[0, 1], format_func=lambda x: "Nie" if x == 0 else "Tak")
    high_blood_pressure = st.selectbox("Nadciśnienie", options=[0, 1], format_func=lambda x: "Nie" if x == 0 else "Tak")
    smoking = st.selectbox("Palenie", options=[0, 1], format_func=lambda x: "Nie" if x == 0 else "Tak")

with col2:
    creatinine_phosphokinase = st.number_input("CPK (mcg/L)",
                                               min_value=int(feature_info['ranges']['creatinine_phosphokinase'][0]),
                                               max_value=int(feature_info['ranges']['creatinine_phosphokinase'][1]))
    ejection_fraction = st.number_input("Frakcja wyrzutowa (%)",
                                        min_value=int(feature_info['ranges']['ejection_fraction'][0]),
                                        max_value=int(feature_info['ranges']['ejection_fraction'][1]))
    platelets = st.number_input("Płytki krwi", min_value=int(feature_info['ranges']['platelets'][0]),
                                max_value=int(feature_info['ranges']['platelets'][1]))
    serum_creatinine = st.number_input("Kreatynina (mg/dL)",
                                       min_value=float(feature_info['ranges']['serum_creatinine'][0]),
                                       max_value=float(feature_info['ranges']['serum_creatinine'][1]), format="%.1f",
                                       step=0.1)
    serum_sodium = st.number_input("Sód (mEq/L)", min_value=int(feature_info['ranges']['serum_sodium'][0]),
                                   max_value=int(feature_info['ranges']['serum_sodium'][1]))
    time = st.number_input("Czas obserwacji (dni)", min_value=int(feature_info['ranges']['time'][0]),
                           max_value=int(feature_info['ranges']['time'][1]))

# Przycisk do predykcji
if st.button("Wykonaj predykcję", type="primary"):
    # Tworzenie danych wejściowych
    input_data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }

    # Konwersja na DataFrame
    input_df = pd.DataFrame([input_data])

    # Upewnienie się, że kolumny są w odpowiedniej kolejności
    input_df = input_df[feature_info['names']]

    # Skalowanie danych
    input_scaled = scaler.transform(input_df)

    # Predykcja
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prediction_proba > 0.5 else 0

    # Wyświetlenie wyniku
    st.markdown("---")
    st.header("Wynik predykcji")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prawdopodobieństwo zgonu")
        st.markdown(
            f"<h1 style='text-align: center; color: {'red' if prediction else 'green'};'>{prediction_proba:.1%}</h1>",
            unsafe_allow_html=True)

        risk_level = "Wysokie" if prediction else "Niskie"
        risk_color = "red" if prediction else "green"
        st.markdown(f"<h3 style='text-align: center; color: {risk_color};'>Ryzyko: {risk_level}</h3>",
                    unsafe_allow_html=True)

    with col2:
        st.subheader("Interpretacja")
        if prediction:
            st.error(
                "Model przewiduje wysokie ryzyko zgonu dla tego pacjenta na podstawie podanych danych klinicznych.")
        else:
            st.success(
                "Model przewiduje niskie ryzyko zgonu dla tego pacjenta na podstawie podanych danych klinicznych.")

        st.info("Pamiętaj, że jest to tylko wskazówka od modelu AI i powinna być zawsze weryfikowana przez lekarza.")

    # Generowanie epikryzy z Gemini
    if GEMINI_AVAILABLE and gemini_model:
        st.markdown("---")
        st.header("Epikryza medyczna")

        with st.spinner("Generowanie epikryzy..."):
            prompt = f"""
            Jako kardiolog, na podstawie danych pacjenta (wiek: {age}, płeć: {'męska' if sex else 'żeńska'}, 
            anemia: {'tak' if anaemia else 'nie'}, CPK: {creatinine_phosphokinase}, cukrzyca: {'tak' if diabetes else 'nie'}, 
            frakcja wyrzutowa: {ejection_fraction}%, nadciśnienie: {'tak' if high_blood_pressure else 'nie'}, 
            płytki krwi: {platelets}, kreatynina: {serum_creatinine}, sód: {serum_sodium}, 
            palenie: {'tak' if smoking else 'nie'}, okres obserwacji: {time} dni) 
            i przewidywania modelu AI (ryzyko zgonu: {'wysokie' if prediction else 'niskie'}, 
            prawdopodobieństwo: {prediction_proba:.1%}), napisz profesjonalną epikryzę medyczną w języku polskim, 
            w następującym schemacie:
            PACJENT ...
            PARAMETRY ŻYCIOWE ...
            WYKONANE BADANIA ...
            ROZPOZNANIE ...
            PRZEWIDYWANY SKUTEK CHOROBY ...
            ZALECENIA ...
            
            Uwaga - nie pisz na końcu uwagi, że jest to przykładowa epikryza, ja o tym wiem. 
            Robię teraz po prostu doświadczenie. Mam świadomość, że to nie są prawdziwe dane, ani zalecenia, nie ostrzegaj mnie o tym.
            """

            try:
                response = gemini_model.generate_content(prompt)
                st.markdown(response.text)

                # Przycisk do pobrania epikryzy
                from datetime import datetime

                current_date = datetime.now().strftime("%Y-%m-%d")
                epicrisis_filename = f"epikryza_{current_date}.txt"
                st.download_button(
                    label="Pobierz epikryzę",
                    data=response.text,
                    file_name=epicrisis_filename,
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Błąd generowania epikryzy: {e}")
    elif GEMINI_AVAILABLE and api_key is None:
        st.markdown("---")
        st.header("Epikryza medyczna")
        st.info("Wprowadź klucz API Gemini w panelu bocznym, aby wygenerować epikryzę.")