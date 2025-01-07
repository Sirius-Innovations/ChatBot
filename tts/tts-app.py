import gradio as gr
from TTS.api import TTS
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

model_name = TTS.list_models()[0]
tts = TTS(model_name)

available_speakers = [
    "Daisy Studious", "Deepika", "Asya Anara",
    "Eugenio Mataracı", "Viktor Menelaos", "Sreenivas"      
]

available_languages = ["US English", "Spanish (LatAm)", "French (fr)", "Sanskrit (hi)"]

selected_speaker = available_speakers[5]
selected_language = available_languages[3]

os.makedirs("output", exist_ok=True)
# Global variable to store the last generated audio path and text
last_generated_audio = None
last_generated_text = ""

def trim_text(text, max_length=30):
    """
    Trim the text to a maximum length and add ellipsis if it exceeds the limit.
    """
    return text[:max_length] + "..." if len(text) > max_length else text


def generate_speech_with_timestamps(text, speaker, language):
    global last_generated_audio, last_generated_text
    
    output_path = "output/generated_speech.wav"
    start_time = time.time()

    speaker_id = "Sofia Hellen" if speaker == "Deepika" else 'Damien Black' if speaker == "Sreenivas" else speaker


  
    tts.tts_to_file(
        text=text,
        speaker=speaker_id,
        #language='en' if language == "US English" else 'es',
        language = 'en' if language == "US English" else 'fr' if language == "French" else 'hi' if language == "Sanskrit" else 'es',
        file_path=output_path
    )
   

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    last_generated_audio = output_path
    last_generated_text = text
    


    samplerate, data = wavfile.read(output_path)
    speech_length = len(data) / samplerate
    
    return output_path, len(text.split()), speaker, language, round(speech_length, 2), duration


# Waveform Function
def generate_waveform(speaker, language):   
    
    global last_generated_audio, last_generated_text

    # Check if a valid audio file exists
    if not last_generated_audio or not os.path.exists(last_generated_audio):
        return None, "No valid audio file found to generate waveform."

    # Read Audio File and Create Time Axis
    samplerate, data = wavfile.read(last_generated_audio)
    time_axis = np.linspace(0, len(data) / samplerate, num=len(data))

    # Plot the Waveform with Custom Styling
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1E1E1E')  # Dark background

    # Plot the Waveform with Custom Styling
    ax.plot(time_axis, data, color='cyan', alpha=0.8, linewidth=1.2)

    # Styling grid and axes for a modern look
    ax.set_facecolor('#2E2E2E')  # Set darker plot background
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # Add grid lines
    ax.spines['bottom'].set_color('white')  # Set bottom spine color to white
    ax.spines['left'].set_color('white')  # Set left spine color to white
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color
    ax.set_xlabel("Time (seconds)", color='white')  # Label x-axis
    ax.set_ylabel("Amplitude", color='white')  # Label y-axis

    # Add a Title to the Plot
    # Trim long text for display in title
    trimmed_text = trim_text(last_generated_text)
   # ax.set_title(f"Waveform for text input: '{trimmed_text}'", color='white', fontsize=14)
    ax.set_title(f"Waveform for '{trimmed_text}'\nSpeaker: {speaker} | Language: {language}", color='white', fontsize=14)

    # Save the waveform image
    waveform_image_path = "output/waveform.png"
    plt.savefig(waveform_image_path, transparent=True)
    plt.close()

    return waveform_image_path, "Waveform generated successfully!"

# Button Click Event Handler
def generate_speech(text, speaker, language):
    if not text:
        return None, "Please enter some text to generate speech.", "", gr.update(interactive=False)

    audio_path, word_count, speaker_name, lang, speech_length, duration = generate_speech_with_timestamps(text, speaker, language)

    # Format the text box content
    data_info = f"Word Count: {word_count}\nVoice: {speaker_name}\nLocalization: {lang}\nLength of Speech: {speech_length} seconds\nGeneration Duration: {duration} seconds"

    return audio_path, data_info, "Speech generation successful!", gr.update(interactive=True)

# Gradio Interface Setup
def setup_interface():
    with gr.Blocks() as app:
       
        gr.Markdown("# 🗣️ Text-to-Speech GenAI")
        gr.Markdown("Convert text to speech using Coqui TTS with support for different languages and speakers.")

        with gr.Row():
            with gr.Column():
                
                text_input = gr.Textbox(label="Enter Text", placeholder="Type your text here...", lines=3)


                with gr.Row():
                   
                    speaker_dropdown = gr.Dropdown(choices=available_speakers, value=selected_speaker, label="Select Voice")
                    language_radio = gr.Radio(choices=available_languages, value=selected_language, label="Select Localization")



            with gr.Column():
               
                data_info_display = gr.Textbox(label="Data Info", interactive=False, lines=5)
                status_message = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            with gr.Column():
               
                audio_output = gr.Audio(label="Generated Speech", interactive=False)
                generate_button = gr.Button("Generate Speech")

            with gr.Column():
               
                waveform_output = gr.Image(label="Waveform")
                generate_waveform_button = gr.Button("Generate Waveform", interactive=False)

        generate_button.click(
            generate_speech, 
            inputs=[text_input, speaker_dropdown, language_radio], 
            outputs=[audio_output, data_info_display, status_message, generate_waveform_button]
        )

        generate_waveform_button.click(
            generate_waveform, 
            inputs=[speaker_dropdown, language_radio], 
            outputs=[waveform_output, status_message]
        )

    return app


if __name__ == "__main__":
    app = setup_interface()
    app.launch(share=True)