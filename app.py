    "Research demonstration only. This app is not a medical diagnostic tool. "
    "A qualified radiologist must interpret MRI scans and make clinical decisions."
)

with st.sidebar:
    st.header("Model settings")
    st.write("Model: EfficientNetB0")
    st.write(f"Decision threshold: {THRESHOLD:.2f}")
    st.caption("The model must be the same binary model used during training.")
    st.caption("Preprocessing assumes RGB, 224 × 224, float32 pixel values in [0, 255].")

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["png", "jpg", "jpeg"],
    help="Use an MRI image compatible with the data used to train the model.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        st.stop()

    st.subheader("Uploaded image")
    st.image(image, caption="Input MRI", width=420)

    if st.button("Analyze MRI", type="primary"):
        try:
            model = load_model(MODEL_PATH)
            batch = preprocess_image(image)

            with st.spinner("Running prediction..."):
                tumour_probability, predicted_label, predicted_score = predict_binary(model, batch)

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted class", predicted_label)
            col2.metric("Predicted-class score", f"{predicted_score * 100:.2f}%")
            col3.metric("Tumour probability", f"{tumour_probability * 100:.2f}%")

            st.subheader("Binary class probabilities")
            st.progress(float(tumour_probability), text=f"Tumourous: {tumour_probability * 100:.2f}%")
            st.progress(
                float(1.0 - tumour_probability),
                text=f"Non-Tumourous: {(1.0 - tumour_probability) * 100:.2f}%",
            )

            st.caption(
                "These are model outputs, not necessarily calibrated clinical probabilities. "
                "A binary model cannot provide separate glioma, meningioma, and pituitary probabilities."
            )

            st.divider()
            st.subheader("Explainable AI: Grad-CAM")
            st.write(
                "Grad-CAM highlights image regions that influenced the selected class score. "
                "It is not a tumour segmentation mask."
            )

            try:
                target_layer, backbone = find_last_conv_layer(model)
                if target_layer is None:
                    st.warning("No Conv2D or DepthwiseConv2D layer was found in this model.")
                else:
                    st.caption(f"Target layer: {target_layer.name}")
                    with st.spinner("Generating Grad-CAM heatmap..."):
                        heatmap = make_gradcam_heatmap(
                            batch,
                            model,
                            target_layer,
                            backbone,
                            tumour_class=(predicted_label == "Tumourous"),
                        )
                        overlay, raw_heatmap = overlay_heatmap(image, heatmap)

                    original_col, overlay_col, raw_col = st.columns(3)
                    with original_col:
                        st.image(image, caption="Original MRI", use_container_width=True)
                    with overlay_col:
                        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                    with raw_col:
                        st.image(raw_heatmap, caption="Grad-CAM heatmap", use_container_width=True)

            except Exception as exc:
                st.warning(f"Prediction succeeded, but Grad-CAM failed: {exc}")
                st.code(
                    "If this continues, share the full terminal traceback and "
                    "the model.summary() output so the target layer can be selected "
                    "for this exact architecture."
                )

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not analyze this image: {exc}")
            st.caption("Check the model path, dependencies, and preprocessing used during training.")

st.divider()
st.caption(
    "Educational research application • Not validated for clinical diagnosis • "
    "Grad-CAM provides qualitative model explanations only"
)
