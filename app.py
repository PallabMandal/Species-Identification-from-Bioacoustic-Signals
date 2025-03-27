import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import wikipedia
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import cv2
import os
from IPython.display import Audio

class BirdSpeciesRecognitionApp:
    def __init__(self):
        # Load pre-trained model directly
        try:
            self.model = load_model('my_model_3.h5')
        except Exception as e:
            st.error(f"Could not load the pre-trained model: {e}")
            self.model = None
        
        # Load species labels
        self.load_species_labels()

    def load_species_labels(self):
        """
        Load species labels from a CSV file
        """
        try:
            labels_df = pd.read_csv('./birdclef-2025/taxonomy.csv')
            self.species_labels = labels_df['common_name'].tolist()  # or 'scientific_name' if needed
        except Exception as e:
            st.error(f"Could not load species labels: {e}")
            self.species_labels = [f'Species {i+1}' for i in range(206)]


    def audio_to_spectrogram(self, audio_file, max_pad_len=64):
        """
        Convert audio to mel spectrogram image
        """
        try:
            # Load audio file
            audio, sample_rate = librosa.load(audio_file, duration=5)
            
            # Generate Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio, 
                sr=sample_rate, 
                n_mels=64,  # Match input shape height
                n_fft=2048, 
                hop_length=512
            )
            
            # Convert to decibel scale
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Normalize
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
            
            # Resize to match model input shape
            mel_image = cv2.resize(mel_spectrogram, (64, 64))
            
            # Add color channel dimension
            mel_image = np.stack((mel_image,)*3, axis=-1)
            
            return mel_image
        except Exception as e:
            st.error(f"Error converting audio to spectrogram: {e}")
            return None

    def predict_species(self, spectrogram):
        """
        Predict bird species from spectrogram
        """
        try:
            # Expand dimensions to match model input
            input_data = np.expand_dims(spectrogram, axis=0)
            
            # Make prediction
            prediction = self.model.predict(input_data)
            
            # Get top 3 predictions
            top_3_indices = prediction[0].argsort()[-3:][::-1]
            top_3_species = [self.species_labels[i] for i in top_3_indices]
            top_3_probabilities = prediction[0][top_3_indices]
            
            return top_3_species, top_3_probabilities
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

    def get_species_info(self, species_name):
        """
        Fetch Wikipedia information about the species
        """
        try:
            # Try to get Wikipedia summary
            page = wikipedia.page(species_name)
            summary = page.summary
            return {
                'summary': summary,
                'url': page.url
            }
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning(f"Multiple matches found. Suggestions: {e.options[:5]}")
            return None
        except Exception as e:
            st.error(f"Could not fetch Wikipedia info: {e}")
            return None

    def record_audio(self, duration=5, sample_rate=44100):
        """
        Record audio from microphone
        """
        st.info(f"Recording audio for {duration} seconds...")
        recording = sd.rec(int(duration * sample_rate), 
                            samplerate=sample_rate, 
                            channels=1, 
                            dtype='float64')
        sd.wait()
        
        # Save recording
        output_filename = 'recorded_audio.wav'
        sf.write(output_filename, recording, sample_rate)
        return output_filename

    def run(self):
        """
        Main Streamlit app
        """
        st.title("🐦 Species Identification from Bioacoustic Signals")
        
        # Sidebar image display
        image_path = "birdclef.png"  # Update with the actual path
        if os.path.exists(image_path):
            st.sidebar.image(image_path, use_column_width=True)
        else:
            st.sidebar.warning("No image available")


        # Sidebar for instructions
        st.sidebar.header("How to Use")
        st.sidebar.info("""
        1. Choose audio input method
        2. Upload or record bird sound
        3. Click 'Identify Species'
        4. View species information
        """)

        image_path = "deer.jpg"  # Update with the actual path
        if os.path.exists(image_path):
            st.sidebar.image(image_path, use_column_width=True)
        else:
            st.sidebar.warning("No image available")

        # Audio input selection
        input_method = st.radio(
            "Select Audio Input Method", 
            [
                "Upload Audio File", 
                # "Record from Microphone"
             ]
        )

        audio_file = None
        if input_method == "Upload Audio File":
            audio_file = st.file_uploader(
                "Upload .ogg or .wav or .mp3 file", 
                type=['ogg', 'wav', 'mp3']
            )
        else:
            if st.button("Start Recording"):
                audio_file = self.record_audio()


        # Save uploaded file temporarily
        if audio_file is not None:

            if isinstance(audio_file, str):  # If it's a file path 
                temp_audio_path = audio_file
            else:  # If it's a recorded audio
                temp_audio_path = "recorded_audio.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.read())

            # didsplay audio player
            # print(audio_file)
            # st.write(Audio(audio_file, rate=22050))

            # Display the uploaded/recorded audio
            st.audio(temp_audio_path, format="audio/wav")

        # # Prediction and display
        # if audio_file is not None:
        #     # Save uploaded file temporarily
        #     with open("recorded_audio.wav", "wb") as f:
        #         f.write(audio_file.read() if hasattr(audio_file, 'read') else audio_file)
            
            # Convert audio to spectrogram
            spectrogram = self.audio_to_spectrogram("recorded_audio.wav")
            
            if spectrogram is not None:
                # Display spectrogram
                st.subheader("Audio Spectrogram")
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(
                    librosa.power_to_db(librosa.feature.melspectrogram(
                        y=librosa.load("recorded_audio.wav")[0]
                    ), 
                    ref=np.max)
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel Spectrogram')
                st.pyplot(plt)
                
                # Predict species
                # if st.button("Identify Species"):
                top_3_species, top_3_probabilities = self.predict_species(spectrogram)
                
                if top_3_species:
                    st.subheader("Top 3 Predicted Species")
                    for species, prob in zip(top_3_species, top_3_probabilities):
                        st.write(f"{species}: {prob*100:.2f}%")
                        
                        # Fetch and display species info for top prediction
                        if species == top_3_species[0]:
                            species_info = self.get_species_info(species)
                            
                            if species_info:
                                st.subheader("Species Information")
                                st.write(species_info['summary'])
                                st.markdown(f"[Read more on Wikipedia]({species_info['url']})")

        # Clean up temporary files
        if os.path.exists("recorded_audio.wav"):
            os.remove("recorded_audio.wav")

def main():
    app = BirdSpeciesRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()

