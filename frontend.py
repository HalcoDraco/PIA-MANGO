import streamlit as st
import os
import tempfile
import zipfile
from settings_manager import SettingsManager
from features.feature_1 import train_dreambooth_model
from features.feature_2 import run_replicate_model
from features.feature_3 import relight_image

settings = SettingsManager.get_settings()
REPLICATE_API_KEY = settings.get("replicate_api_key", None)
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

st.set_page_config(page_title="AI Studio", layout="wide")

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Logo_of_Mango_%28new%29.svg/1280px-Logo_of_Mango_%28new%29.svg.png" style="width:200px; border-radius: 0px;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.header("AI Studio")
    page = st.selectbox("Navigation", ["Home", "Feature 1", "Feature 2", "Feature 3", "Pipeline 1", "Pipeline 2"])

# Home Page
if page == "Home":
    st.title("AI Studio – Elevating Product Imagery with AI")
    st.markdown("""
    AI Studio offers a powerful suite of AI tools to enhance, edit, and generate high-quality product visuals. Users can:
    - Generate diverse and realistic images using the top-5 text-to-image models
    - Restore and edit images precisely with In-Painting powered by LoRA
    - Replace backgrounds and apply intelligent relighting for improved presentation
    - Edit specific image regions via text or reference input
    - Enhance facial features and skin quality in AI-generated images
    
    All in an intuitive, user-friendly interface tailored for creative teams and visual content professionals.
    """)
    st.image("images/banner.jpeg", use_container_width=True)

else:
    # Feature 1
    if page == "Feature 1":
        st.header("Feature 1: Smart Image Restoration with LoRA Technology")
        st.markdown("Deliver unmatched image consistency and precision with our cutting-edge integration of pretrained models and LoRA-based In-Painting. This feature ensures seamless restoration and enhancement, producing visually coherent and detail-rich results every time.")
        
        default_steps = settings["lora_dreambooth_params"]["steps"]
        default_rank = settings["lora_dreambooth_params"]["lora_rank"]
        trained_models = list(settings["models"]["feature_1"].keys())

        tab1, tab2 = st.tabs(["Training", "Inference"])

        # ----------------------- TRAINING -----------------------
        with tab1:
            st.header("Train a LoRA Model")

            model_name = st.text_input("Model Name", value="my-lora-model")
            trigger_word = st.text_input("Trigger Word", value="TOK")
            description = st.text_area("Model Description", "Fine-tuning Flux with LoRA using DreamBooth")
            steps = st.number_input("Training Steps", min_value=100, step=100, value=default_steps)
            lora_rank = st.number_input("LoRA Rank", min_value=4, step=4, value=default_rank)

            uploaded_files = st.file_uploader("Upload Training Images", type=["jpg", "jpeg", "png","zip"], accept_multiple_files=True)

            if st.button("Start Training"):
                if not uploaded_files:
                    st.warning("Please upload at least one image.")
                else:
                    with st.spinner("Uploading and starting training..."):
                        try:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                image_paths = []
                                for idx, uploaded_file in enumerate(uploaded_files):

                                    # Check if the uploaded file is a zip file
                                    if uploaded_file.name.endswith(".zip"):
                                        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                                            zip_ref.extractall(os.path.join(tmpdir, uploaded_file.name, f"{idx}"))
                                            for file in os.listdir(os.path.join(tmpdir, uploaded_file.name, f"{idx}")):
                                                if file.endswith((".jpg", ".jpeg", ".png")):
                                                    image_paths.append(os.path.join(tmpdir, uploaded_file.name, f"{idx}", file))

                                    # Otherwise, treat it as an image file
                                    else:
                                        path = os.path.join(tmpdir, f"img_{idx}.jpg")
                                        with open(path, "wb") as f:
                                            f.write(uploaded_file.read())
                                        image_paths.append(path)

                                zip_path = os.path.join(tmpdir, "images.zip")
                                with zipfile.ZipFile(zip_path, "w") as zf:
                                    for path in image_paths:
                                        zf.write(path, arcname=os.path.basename(path))

                                training = train_dreambooth_model(
                                    model_name=model_name,
                                    images_path=zip_path,
                                    trigger_word=trigger_word,
                                    description=description,
                                    steps=steps,
                                    lora_rank=lora_rank
                                )

                            st.success("Training started!")
                            st.write(f"Training ID: `{training.id}`")
                        except Exception as e:
                            st.error(f"Training failed: {e}")

        # ----------------------- INFERENCE -----------------------
        with tab2:
            st.header("Inference with Trained Model")

            selected_model = st.selectbox("Select Trained Model", options=trained_models)
            model_meta = settings["models"]["feature_1"].get(selected_model, {})
            default_trigger = model_meta.get("trigger_word", "")

            prompt = st.text_area("Prompt", f"A photo of a {default_trigger} in a forest, 8k, photorealistic")

            if st.button("Generate Image"):
                with st.spinner("Generating image..."):
                    try:
                        output_images = run_replicate_model(selected_model, {"prompt": prompt})
                        for img in output_images:
                            st.image(img)
                    except Exception as e:
                        st.error(f"Error generating image: {e}")


    # Feature 2
    elif page == "Feature 2":
        st.header("Feature 2: Premium Text-to-Image Generation Suite")
        st.markdown("Unleash creativity with access to the top 5 industry-leading text-to-image generation models. This powerful suite offers users a wide range of stylistic options and top-tier visual quality, all within a single streamlined interface.")
        feature2_models = settings["models"]["feature_2"]
        model_display_names = {k: f'{v["name"]} ({v["author"]})' for k, v in feature2_models.items()}
        model_key = st.selectbox("Choose a text-to-image model", options=list(model_display_names.keys()), format_func=lambda x: model_display_names[x])
        prompt = st.text_area("Enter your prompt", value="A futuristic fashion ad with a model under neon lights")
        generate_btn = st.button("Generate Image")
        if generate_btn:
            with st.spinner("Generating..."):
                input_data = {"prompt": prompt}
                try:
                    images = run_replicate_model(model_key, input_data)
                    st.success("Images generated successfully!")
                    for img in images:
                        st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to generate image: {str(e)}")

    # Feature 3
    elif page == "Feature 3":
        st.header("Feature 3: AI-Powered Background Replacement & Dynamic Relighting")
        st.markdown("Transform product visuals effortlessly. Our IC-Lighting technology intelligently replaces backgrounds and simulates natural light sources to enhance realism, helping your images stand out across platforms.")
        with st.sidebar:
            st.header("Parameters")
            height = st.number_input("Height", min_value=1, value=640, step=64)
            width = st.number_input("Width", min_value=1, value=512, step=64)
            light = st.selectbox("Light source", ["Left Light", "Right Light", "Top Light", "Bottom Light"])
            n_images = st.number_input("Image number", min_value=1, value=1, step=1)
            steps = st.number_input("Steps", min_value=1, value=25, step=1)
            highres_scale = st.number_input("Highres scale", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        
        uploaded_file = st.file_uploader("Upload image for training", type=["jpg", "jpeg", "png"])
        caption = st.text_input("Enter your prompt for the relighting process", value='Move the sunset light to the right.')
        new_filename = st.text_input("Filename of the generated image", value="new_image.png")
        if st.button("Generate Image"):
            if uploaded_file:
                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_image_path = os.path.join(tmpdir, "input.jpg")
                    with open(temp_image_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    with open(temp_image_path, "rb") as image_file:
                        relight_image(None, image_file, prompt=caption, height=height, width=width, light_source=light, n_images=n_images, highres_scale=highres_scale, steps=steps)
                    if os.path.exists(temp_image_path):
                        st.image('images/output/output_0.jpg', use_container_width=True)
                    else:
                        st.error("Image generation failed. No output file found.")
            else:
                st.warning("⚠️ Please upload an image before generating.")

    # Pipeline 1
    elif page == "Pipeline 1":
        st.header("Pipeline 1: Image Editing via Text or Reference Image")
        st.markdown("Coming soon: Empower users to edit specific image regions using natural language descriptions or example visuals. This intuitive editing pipeline combines In-Painting and IP-Adapter technologies for precision-driven, user-guided transformations.")
        
    # Pipeline 2
    elif page == "Pipeline 2":
        st.header("Pipeline 2: AI-Based Skin and Face Enhancement")
        st.markdown("Coming soon: Achieve flawless, photorealistic results with our AI-based skin and face enhancement pipeline. Specially designed for generated images, it boosts clarity, smoothness, and realism—perfect for editorial and e-commerce applications.")
