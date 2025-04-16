import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import wikipedia
import requests
from bs4 import BeautifulSoup
import tensorflow as tf

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# import sounddevice as sd
# import soundfile as sf
import matplotlib.pyplot as plt
import cv2
import os

# from IPython.display import Audio
# import folium
# from streamlit_folium import folium_static


class BirdSpeciesRecognitionApp:
    def __init__(self):
        # Load pre-trained model directly
        try:
            self.model = load_model("./assets/my_model_3.h5", compile=False)
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
            labels_df = pd.read_csv("taxonomy.csv")
            self.species_labels = labels_df[
                "common_name"
            ].tolist()  # or 'scientific_name' if needed
        except Exception as e:
            st.error(f"Could not load species labels: {e}")
            self.species_labels = [f"Species {i+1}" for i in range(206)]

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
                hop_length=512,
            )

            # Convert to decibel scale
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Normalize
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (
                mel_spectrogram.max() - mel_spectrogram.min()
            )

            # Resize to match model input shape
            mel_image = cv2.resize(mel_spectrogram, (64, 64))

            # Add color channel dimension
            mel_image = np.stack((mel_image,) * 3, axis=-1)

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
        Fetch Wikipedia information, excluding the range map.
        """
        try:
            # st.write(species_name)
            page = wikipedia.page(species_name, auto_suggest=False)
            # limit summary to 4 sentences
            summary = page.summary.split(".")[:5]
            summary = (".".join(summary) + ".").strip()
            # summary = page.summary
            url = page.url

            # Extract image using BeautifulSoup
            response = requests.get(page.url)
            soup = BeautifulSoup(response.text, "html.parser")
            infobox = soup.find("table", {"class": "infobox"})

            image_url = None
            if infobox:
                img_tags = infobox.find_all("img")
                print(len(img_tags))
                if img_tags:
                    # print all img tags
                    image_urls = []
                    for i in range(len(img_tags)):
                        image_url = "https:" + img_tags[i]["src"]
                        print(image_url)
                        image_urls.append(image_url)

            return {"summary": summary, "url": url, "image_urls": image_urls}
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning(f"Multiple matches found. Suggestions: {e.options[:5]}")
            return None
        except Exception as e:
            st.error(f"Could not fetch Wikipedia info: {e}")
            return None

    def run(self):
        """
        Main Streamlit app
        """
        st.title("🐦 Species Identification from Bioacoustic Signals")

        # Sidebar image display
        image_path = "./assets/birdclef.png"  # Update with the actual path
        if os.path.exists(image_path):
            st.sidebar.image(image_path, use_container_width=True)
        else:
            st.sidebar.warning("No image available")

        # Sidebar for instructions
        st.sidebar.header("How to Use")
        st.sidebar.info(
            """
        1. Choose audio input method
        2. Upload or record bird sound
        3. View species information
        """
        )

        image_path = "./assets/deer.jpg"  # Update with the actual path
        if os.path.exists(image_path):
            st.sidebar.image(image_path, use_container_width=True)
        else:
            st.sidebar.warning("No image available")

        # Audio input selection
        input_method = st.radio(
            "Select Audio Input Method",
            [
                "Upload Audio File",
                #   "Record from Microphone"
            ],
        )

        audio_file = None
        if input_method == "Upload Audio File":
            audio_file = st.file_uploader(
                "Upload .ogg or .wav or .mp3 file", type=["ogg", "wav", "mp3"]
            )
        else:
            if st.button("Start Recording"):
                audio_file = self.record_audio()

        # Save uploaded file temporarily
        if audio_file is not None:

            if isinstance(audio_file, str):  # If it's a file path
                temp_audio_path = audio_file
            else:  # If it's audio
                temp_audio_path = "recorded_audio.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.read())

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
                # st.subheader("Audio Spectrogram")
                plt.figure(figsize=(10, 4))  # Extend width, reduce height
                librosa.display.specshow(
                    librosa.power_to_db(
                        librosa.feature.melspectrogram(
                            y=librosa.load("recorded_audio.wav")[0]
                        ),
                        ref=np.max,
                    )
                )
                plt.colorbar(format="%+2.0f dB")
                plt.title("Mel Spectrogram")
                plt.axis("off")  # Remove axes for a cleaner look
                expander = st.expander("Audio Spectrogram", expanded=False)
                with expander:
                    st.pyplot(plt, bbox_inches="tight")
                plt.close()
                # st.pyplot(plt, bbox_inches='tight')

                # Predict species
                # if st.button("Identify Species"):
                top_3_species, top_3_probabilities = self.predict_species(spectrogram)

                if top_3_species:
                    for species, prob in zip(top_3_species, top_3_probabilities):
                        if species == top_3_species[0]:  # Top prediction
                            st.markdown(
                                f"### Species inferred: {species}"
                            )  # H3 header with bold effect
                            # else:
                            #     st.write(f"Species inferred: ***{species}***", unsafe_allow_html=True)

                            # if species == top_3_species[0]:  # Only fetch info for top prediction
                            species_info = self.get_species_info(species)

                            if species_info:
                                st.subheader("📖 Species Information")
                                st.write(species_info["summary"] + "\n")

                                # Display all images in image_urls list side by side
                                if species_info["image_urls"]:
                                    # st.subheader("Images")
                                    cols = st.columns(2)  # Create two columns
                                    with cols[0]:
                                        st.image(
                                            species_info["image_urls"][0],
                                            caption=f"**{species}**",
                                            use_container_width=True,
                                        )
                                    map_url = ""
                                    for img_url in species_info["image_urls"]:
                                        # If img_url contains "map", assign and break
                                        if "map" in img_url:
                                            map_url = img_url
                                            break
                                    if map_url:
                                        with cols[1]:
                                            st.image(
                                                map_url,
                                                caption=f"**🌍 Habitat Range**",
                                                use_container_width=True,
                                            )
                                    else:
                                        with cols[1]:
                                            st.image(
                                                species_info["image_urls"][-1],
                                                use_container_width=True,
                                            )

                                # Link to full Wikipedia article
                                st.markdown(
                                    f"[🔗 Read more on Wikipedia]({species_info['url']})"
                                )

        # Clean up temporary files
        if os.path.exists("recorded_audio.wav"):
            os.remove("recorded_audio.wav")


def main():
    app = BirdSpeciesRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()
