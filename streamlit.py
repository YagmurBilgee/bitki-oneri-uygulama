import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bitki Oneri Sistemi", layout="wide")

st.title("Bitki Oneri Sistemi")
st.markdown("Toprak ve iklim ozelliklerine gore en uygun bitki onerisini alabilirsiniz.")

@st.cache_resource
def load_model():
     model = joblib.load("crop_model_rf.pkl")
    scaler = joblib.load("crop_scaler.pkl")
    

    with open("model/model_info.txt", "r") as f:
        lines = f.readlines()
        model_info = {
            "model_name": lines[0].split(": ")[1].strip(),
            "feature_selection": lines[1].split(": ")[1].strip(),
            "features": lines[2].split(": ")[1].strip().split(", "),
            "f1_score": float(lines[3].split(": ")[1].strip())
        }
    
    return model, scaler, model_info


try:
    model, scaler, model_info = load_model()
    st.success(f"Model basariyla yuklendi: {model_info['model_name']}")
    st.info(f"Model performansi (F1-score): {model_info['f1_score']:.4f}")
    
    
    with st.expander("Model detaylari"):
        st.write(f"Ozellik secim yontemi: {model_info['feature_selection']}")
        st.write(f"Kullanilan ozellikler: {', '.join(model_info['features'])}")
    
    
    feature_descriptions = {
        "N": "Azot (N) - kg/ha",
        "P": "Fosfor (P) - kg/ha",
        "K": "Potasyum (K) - kg/ha",
        "temperature": "Sicaklik - Celsius",
        "humidity": "Nem - %",
        "ph": "Toprak pH degeri",
        "rainfall": "Yagis - mm"
    }
    
    
    st.sidebar.header("Toprak ve Iklim Ozellikleri")
    
    user_input = {}
    for feature in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:

        if feature == "N":
            min_val, max_val, default_val = 0, 140, 70
        elif feature == "P":
            min_val, max_val, default_val = 5, 145, 50
        elif feature == "K":
            min_val, max_val, default_val = 5, 205, 40
        elif feature == "temperature":
            min_val, max_val, default_val = 8, 44, 25
        elif feature == "humidity":
            min_val, max_val, default_val = 14, 100, 65
        elif feature == "ph":
            min_val, max_val, default_val = 3.5, 10.0, 6.5
            step = 0.1
        elif feature == "rainfall":
            min_val, max_val, default_val = 20, 300, 100
        
        if feature == "ph":
            user_input[feature] = st.sidebar.slider(
                feature_descriptions[feature], 
                min_value=min_val, 
                max_value=max_val, 
                value=default_val,
                step=0.1
            )
        else:
            user_input[feature] = st.sidebar.slider(
                feature_descriptions[feature], 
                min_value=min_val, 
                max_value=max_val, 
                value=default_val
            )
    
  
    if st.sidebar.button("Bitki Onerisi Al"):
        
        input_df = pd.DataFrame([user_input])
        
        input_scaled = scaler.transform(input_df)
        
        features_to_use = model_info['features']
        
    
        feature_indices = [list(input_df.columns).index(f) for f in features_to_use]
        
        input_selected = input_scaled[:, feature_indices]
        
        prediction = model.predict(input_selected)[0]
        
        
        st.markdown("## Oneri Sonucu")
        st.success(f"Onerilen Bitki: **{prediction}**")
        
        try:
            probabilities = model.predict_proba(input_selected)[0]
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            
            st.markdown("### En Olasi 3 Bitki")
            for idx in top_3_idx:
                crop = model.classes_[idx]
                prob = probabilities[idx]
                st.write(f"{crop}: {prob*100:.2f}%")
        except:
            st.info("Bu model olasilik hesaplamalarini desteklemiyor.")
        
        
        st.markdown("### Girilen Degerler")
        st.dataframe(input_df)
        
except Exception as e:
    st.error(f"Model yuklenemedi: {e}")
    st.warning("Lutfen once 'test.py' dosyasini calistirip modeli kaydedin.") 