from __future__ import annotations

from io import BytesIO

import requests
import streamlit as st
from PIL import Image

from src.config import settings

st.set_page_config(page_title="Skin Disease Detection Demo", layout="centered")
st.title("Skin Disease Detection & LLM Advisor")
st.caption("Upload a skin image, send it to the FastAPI backend, and view the prediction with recommendations.")

api_url = st.sidebar.text_input("FastAPI URL", value=f"{settings.api_base_url}/analyze_skin")
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file is not None:
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze Skin"):
        with st.spinner("Analyzing..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "image/jpeg",
                    )
                }
                response = requests.post(api_url, files=files, timeout=120)
                response.raise_for_status()
                result = response.json()

                st.subheader("Prediction")
                st.write(f"**Disease:** {result['disease']}")
                st.write(f"**Confidence:** {result['confidence']:.2%}")

                st.subheader("Recommendations")
                st.write(result["recommendations"])

                st.subheader("Next Steps")
                st.write(result["next_steps"])

                st.subheader("Tips")
                st.write(result["tips"])

                with st.expander("Raw API Response"):
                    st.json(result)
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
