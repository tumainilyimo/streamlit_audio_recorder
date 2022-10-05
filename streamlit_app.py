# streamlit_audio_recorder by stefanrmmr (rs. analytics) - version April 2022

import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components


# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="streamlit_audio_recorder")
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
    unsafe_allow_html=True)
# Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
    unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
    unsafe_allow_html=True)  # lightmode


def audiorec_demo_app():

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Custom REACT-based component for recording client audio in browser
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    # specify directory and initialize st_audiorec object functionality
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    # TITLE and Creator information
    st.title('streamlit audio recorder')
    st.markdown('Implemented by '
        '[Stefan Rummer](https://www.linkedin.com/in/stefanrmmr/) - '
        'view project source code on '
        '[GitHub](https://github.com/stefanrmmr/streamlit_audio_recorder)')
    st.write('\n\n')


    # STREAMLIT AUDIO RECORDER Instance
    val = st_audiorec()
    # web component returns arraybuffer from WAV-blob
    st.write('Audio data received in the Python backend will appear below this message ...')

    if isinstance(val, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            val = np.array(val)             # convert to np array
            sorted_ints = val[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()

        # wav_bytes contains audio data in format to be further processed
        # display audio data as received on the Python side
        st.audio(wav_bytes, format='audio/wav')
import io
import whisper
import torch
import ffmpeg
import torchaudio
import streamlit as st
LANGUAGES = {
    "en":"english",
    "zh":"chinese",
    "de":"german",
    "es":"spanish",
    "ru":"russian",
    "ko":"korean",
    "fr":"french",
    "ja":"japanese",
    "pt":"portuguese",
    "tr":"turkish",
    "pl":"polish",
    "ca":"catalan",
    "nl":"dutch",
    "ar":"arabic",
    "sv":"swedish",
    "it":"italian",
    "id":"indonesian",
    "hi":"hindi",
    "fi":"finnish",
    "vi":"vietnamese",
    "iw":"hebrew",
    "uk":"ukrainian",
    "el":"greek",
    "ms":"malay",
    "cs":"czech",
    "ro":"romanian",
    "da":"danish",
    "hu":"hungarian",
    "ta":"tamil",
    "no":"norwegian",
    "th":"thai",
    "ur":"urdu",
    "hr":"croatian",
    "bg":"bulgarian",
    "lt":"lithuanian",
    "la":"latin",
    "mi":"maori",
    "ml":"malayalam",
    "cy":"welsh",
    "sk":"slovak",
    "te":"telugu",
    "fa":"persian",
    "lv":"latvian",
    "bn":"bengali",
    "sr":"serbian",
    "az":"azerbaijani",
    "sl":"slovenian",
    "kn":"kannada",
    "et":"estonian",
    "mk":"macedonian",
    "br":"breton",
    "eu":"basque",
    "is":"icelandic",
    "hy":"armenian",
    "ne":"nepali",
    "mn":"mongolian",
    "bs":"bosnian",
    "kk":"kazakh",
    "sq":"albanian",
    "sw":"swahili",
    "gl":"galician",
    "mr":"marathi",
    "pa":"punjabi",
    "si":"sinhala",
    "km":"khmer",
    "sn":"shona",
    "yo":"yoruba",
    "so":"somali",
    "af":"afrikaans",
    "oc":"occitan",
    "ka":"georgian",
    "be":"belarusian",
    "tg":"tajik",
    "sd":"sindhi",
    "gu":"gujarati",
    "am":"amharic",
    "yi":"yiddish",
    "lo":"lao",
    "uz":"uzbek",
    "fo":"faroese",
    "ht":"haitian creole",
    "ps":"pashto",
    "tk":"turkmen",
    "nn":"nynorsk",
    "mt":"maltese",
    "sa":"sanskrit",
    "lb":"luxembourgish",
    "my":"myanmar",
    "bo":"tibetan",
    "tl":"tagalog",
    "mg":"malagasy",
    "as":"assamese",
    "tt":"tatar",
    "haw":"hawaiian",
    "ln":"lingala",
    "ha":"hausa",
    "ba":"bashkir",
    "jw":"javanese",
    "su":"sundanese",
}

def decode(model, mel, options):
    result = whisper.decode(model, mel, options)
    return result.text

def load_audio(audio):
    print(audio.type)
    if audio.type == "audio/wav" or audio.type == "audio/flac":
        wave, sr = torchaudio.load(audio)
        if sr != 16000:
            wave = torchaudio.transforms.Resample(sr, 16000)(wave)
        return wave.squeeze(0)

    elif audio.type == "audio/mpeg":
        audio = audio.read()
        audio, _ = (ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16k')
            .run(capture_stdout=True, input=audio)
        )
        audio = io.BytesIO(audio)
        wave, sr = torchaudio.load(audio)
        if sr != 16000:
            wave = torchaudio.transforms.Resample(sr, 16000)(wave)
        return wave.squeeze(0)

    else:
        st.error("Unsupported audio format")

def detect_language(model, mel):
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)

def main():

    st.title("Whisper ASR Demo")
    st.markdown(
            """
        This is a demo of OpenAI's Whisper ASR model. The model is trained on 680,000 hours of dataset. 
        """
    )

    model_selection = st.sidebar.selectbox("Select model", ["tiny", "base", "small", "medium", "large"])
    en_model_selection = st.sidebar.checkbox("English only model", value=False)

    if en_model_selection:
        model_selection += ".en"
    st.sidebar.write(f"Model: {model_selection+' (Multilingual)' if not en_model_selection else model_selection + ' (English only)'}")

    if st.sidebar.checkbox("Show supported languages", value=False):
            st.sidebar.info(list(LANGUAGES.values()))
    st.sidebar.title("Options")
    
    beam_size = st.sidebar.slider("Beam Size", min_value=1, max_value=10, value=5)
    fp16 = st.sidebar.checkbox("Enable FP16 for faster transcription (It may affect performance)", value=False)

    if not en_model_selection:
        task = st.sidebar.selectbox("Select task", ["transcribe", "translate (To English)"], index=0)
    else:
        task = st.sidebar.selectbox("Select task", ["transcribe"], index=0)

    st.title("Audio")
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])

    if audio_file is not None:
        st.audio(audio_file, format=audio_file.type)
        with st.spinner("Loading model..."):
            model = whisper.load_model(model_selection)
            model = model.to("cpu") if not torch.cuda.is_available() else model.to("cuda")
            

        audio = load_audio(audio_file)
        with st.spinner("Extracting features..."):
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
        if not en_model_selection:
            with st.spinner("Detecting language..."):
                language = detect_language(model, mel)
                st.markdown(f"Detected Language: {LANGUAGES[language]} ({language})")
        else:
            language = "en"
        configuration = {"beam_size": beam_size, "fp16": fp16, "task": task, "language": language}
        with st.spinner("Transcribing..."):
            options = whisper.DecodingOptions(**configuration)
            text = decode(model, mel, options)
        st.markdown(f"**Recognized Text:** {text}")



    import openai as ai

    def chat(question,chat_log = None) -> str:
        if(chat_log == None):
            chat_log = start_chat_log
        prompt = f"{chat_log}Human: {question}\nAI:"
        response = completion.create(prompt = prompt, engine =  "text-davinci-002", temperature = 0.7,top_p=1, frequency_penalty=0, 
        presence_penalty=0.7, best_of=2,max_tokens=1000,stop = "\nHuman: ")
        return response.choices[0].text
    
    def modify_start_message(chat_log,question,answer) -> str:
        if chat_log == None:
            chat_log = start_chat_log
        chat_log += f"Human: {question}\nAI: {answer}\n"
        return chat_log
    
    if __name__ == "__main__":
        ai.api_key = "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"
    
        completion = ai.Completion()
    
        start_chat_log = """Human: nini maana ya viazi lishe?.
        AI: Viazi lishe ni aina ya viazi vitamu vyenye mwonekano wa rangi ya njano kwa ndani vikimenywa. Viazi lishe ni biashara mpya katika 
        kilimo biashara ambacho hakijamulikwa vyema na wakulima wetu. Hii ni fursa muhimu na ya kipekee kwa wakulima wanaolenga utajiri 
        kutokana na kilimo. Kwa kifupi huku kuna pesa kuliko kawaida. Ni muhimu tukaichangamkia fursa hii. Kilimo cha viazi lishe ni kama 
        ilivyo kwa viazi vingine vitamu ni rahisi sana ukilinganisha na aina nyingine ya kilimo..
        Human: How are you?
        AI: I am fine, thanks for asking.
        Human: what is your name
        AI: I am Aflatoxin A.I Chartbot
        
        Human:how old are you?
        AI: Sorry i cant say that
        """
    #question = ""
    print("\nEnter the questions to  Codebot (to quit type \"stop\")")
    
    
    
    #while True:
    
    #question = (f"**Recognized Text:** {text}")
    #while True:
    question = (f"**Recognized Text:** {text}")
       
    st.write("Codebot:" ,chat(question,start_chat_log))

if __name__ == '__main__':

    # call main function
    audiorec_demo_app()
