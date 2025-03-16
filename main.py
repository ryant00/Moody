import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import streamlit as st
import time

st.set_page_config(
    page_title="MoodyAI",  # Nama tab
    page_icon="MOODY LOGO.png"
)


#with st.spinner("Loading... Please wait"):
#st.image("MOODY LOGO.png", use_container_width=True, output_format="PNG", caption="")
   #time.sleep(1)  # Simulasi loading selama 3 detik

# Custom CSS untuk tampilan mobile-like
st.markdown("""
    <style>
    /* Base styling */
.stApp {
    background: linear-gradient(135deg, rgb(180, 224, 208) 35%, rgb(240, 244, 255) 50%, rgb(223, 215, 255) 80%);
    backdrop-filter: blur(100px);
}


    
.block-container {
    max-width: 450px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    background: transparent !important; /* Pastikan latar belakang transparan */
    box-shadow: none !important; /* Hilangkan shadow */
    border: none !important; /* Hilangkan border */
}

    
    /* Phone frame styling */
    .phone-container {
        max-width: 380px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 40px;
        background: transparent;
    }
    
    /* Header styling */
    .header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6b4ce6;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'SF Pro Display', sans-serif;
    }
    
    /* Input box styling */
    .stTextArea textarea {
    background: rgb(131, 101, 252);
    backdrop-filter: blur(10px);
    border: 2px solid rgb(131, 101, 252); /* Warna border disamakan */
    border-radius: 0px;
    padding: 1rem;
    font-size: 1.1rem;
    color: white;
    min-height: 150px;
    outline: none; /* Hilangkan outline saat aktif */
    box-shadow: none; /* Hilangkan shadow tambahan */
    overflow: hidden; /* Hindari overflow yang mungkin menyebabkan ujung hitam */
    }

    
    /* Button styling */
    .stButton button {
        background: #6b4ce6;
        color: white;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        float: left;
    }
    
/* Mood result card */
.mood-card {
    background: rgba(107, 90, 230, 0.7);
    backdrop-filter: blur(100px);
    padding: 1.5rem;
    border-radius: 20px;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease-in-out; /* Transisi yang lebih smooth */
}

.mood-emoji {
    font-size: 2rem;
    margin-right: 0.5rem;
}

/* Efek hover */
.mood-card:hover {
    transform: translateY(-5px) scale(1.05); /* Naik sedikit & membesar */
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5); /* Shadow lebih dalam */
}

    
/* Tips card */
.tips-card {
    background: #b8dcd4;
    padding: 1.5rem;
    border-radius: 20px;
    margin: 1rem 0;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}

/* Efek hover */
.tips-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
}

    
    /* Navigation dots */
    .nav-dots {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .nav-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(107, 76, 230, 0.3);
    }
    
    .nav-dot.active {
        background: #6b4ce6;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #6b4ce6;
        border-radius: 10px;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Model initialization
model = load_model("mood_detection_model.keras")
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class TextPreprocessor:
    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def predict_mood(text):
    cleaned_text = TextPreprocessor.clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)[0]
    
    mood_labels = label_encoder.classes_
    mood_index = np.argmax(prediction).item()
    mood = mood_labels[mood_index]
    
    return {
        'mood': mood,
        'confidence': float(prediction[mood_index]),
        'all_probabilities': dict(zip(mood_labels, prediction.tolist()))
    }

def predict_mood(text):
    cleaned_text = TextPreprocessor.clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)[0]
    
    mood_labels = label_encoder.classes_
    mood_index = np.argmax(prediction).item()
    mood = mood_labels[mood_index]
    
    return {
        'mood': mood,
        'confidence': float(prediction[mood_index]),
        'all_probabilities': dict(zip(mood_labels, prediction.tolist()))
    }

def get_mood_emoji(mood):
    return {
        'happy': '😊',
        'angry': '😠',
        'fear': '😰',
        'sad': '😢'
    }.get(mood, '🤔')

def get_mood_tips(mood):
    tips_data = {
        "happy": {
            "title": "😊 Keep the Good Vibes!",
            "tips": ["Share your happiness with others", "Write down what made you happy today", "Keep up with activities you enjoy"]
        },
        "sad": {
            "title": "😢 Stay Strong!",
            "tips": ["Listen to uplifting music", "Talk to a friend or loved one", "Engage in activities you enjoy"]
        },
        "angry": {
            "title": "😡 Stay Calm!",
            "tips": ["Take deep breaths", "Practice relaxation techniques", "Engage in a hobby"]
        },
        "fear": {
            "title": "😨 Overcome Your Fears!",
            "tips": ["Challenge negative thoughts", "Practice mindfulness", "Talk to someone you trust"]
        }
    }

    # Pastikan selalu mengembalikan dictionary dengan title dan tips
    return tips_data.get(mood, {"title": "No Title", "tips": ["No tips available."]})



def main():
    
    st.markdown('<div class="phone-container">', unsafe_allow_html=True)
    
    # Initial view
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    if st.session_state.step == 1:
        st.markdown('<div class="header">How\'s Your Day?</div>', unsafe_allow_html=True)
        with st.form(key='mood_form'):
            text_input = st.text_area("", placeholder="Tell me something...")
            submit = st.form_submit_button("→")
            
            if submit and text_input:
                st.session_state.text = text_input
                st.session_state.result = predict_mood(text_input)
                st.session_state.step = 2
                st.rerun()
    
    elif st.session_state.step == 2:
        st.markdown('<div class="header">I\'ve Detect Your Mood!</div>', unsafe_allow_html=True)
        result = st.session_state.result
        
        with st.container():
            st.markdown(
                f'<div class="mood-card">'
                f'<span class="mood-emoji">{get_mood_emoji(result["mood"])}</span>'
                f'<span style="font-size: 1.5rem; font-weight: 600;">{result["mood"].upper()}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            for mood, prob in result['all_probabilities'].items():
                st.markdown(f"""<p style="color: black; font-weight: bold;">{mood.title()}: {prob:.1%}</p>""", unsafe_allow_html=True)
                st.progress(prob)
        
        if st.button("Let Me Give You Some Tips →"):
            st.session_state.step = 3
            st.rerun()
    
    elif st.session_state.step == 3:
        st.markdown('<div class="header">Let Me Help You</div>', unsafe_allow_html=True)
        result = st.session_state.result
        tips = get_mood_tips(result['mood'])
        
        st.markdown(
            f'<div class="mood-card">'
            f'<span class="mood-emoji">{get_mood_emoji(result["mood"])}</span>'
            f'<span style="font-size: 1.2rem; font-weight: 600;">{tips["title"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        for tip in tips['tips']:
            st.markdown(f'<div class="tips-card" style="background-color:rgb(1, 172, 135)">{tip}</div>', unsafe_allow_html=True)
        
        if st.button("← Start Over"):
            st.session_state.step = 1
            st.rerun()
    
    # Navigation dots
    st.markdown(
        f'<div class="nav-dots">'
        f'<div class="nav-dot {"active" if st.session_state.step == 1 else ""}"></div>'
        f'<div class="nav-dot {"active" if st.session_state.step == 2 else ""}"></div>'
        f'<div class="nav-dot {"active" if st.session_state.step == 3 else ""}"></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()