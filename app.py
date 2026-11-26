                + probability_bar("Non-Tumourous", res["non_tumour_probability"], GREEN),
                unsafe_allow_html=True,
            )
            st.write(f"Tumourous: {res['tumour_probability'] * 100:.2f}%")
            st.write(f"Non-Tumourous: {res['non_tumour_probability'] * 100:.2f}%")

            if 0.35 <= res["tumour_probability"] <= 0.65:
                st.warning(
                    "The model is close to the decision boundary for this image. "
                    "Treat this prediction with extra caution."
                )

            c1, c2 = st.columns(2)
            c1.metric("Decision threshold", f"{threshold:.2f}")
            c2.metric("Inference time", f"{analysis['seconds']:.1f} s")

            st.caption(
                "Scores are model outputs, not calibrated clinical probabilities, "
                "and do not prove the prediction is correct."
            )

    # ---- Grad-CAM (full width) ----
    if has_result:
        st.divider()
        st.subheader("3 · Explainable AI: Grad-CAM")

        if analysis["gradcam_error"]:
            st.warning(f"Grad-CAM could not be generated: {analysis['gradcam_error']}")
        else:
            res = build_result(analysis["tumour_probability"], threshold)
            heatmap = analysis["heat_tumour"] if res["is_tumour"] else analysis["heat_clear"]
            colored, overlay = overlay_heatmap(image, heatmap, overlay_alpha)

            target_class = "Tumourous" if res["is_tumour"] else "Non-Tumourous"
            st.caption(
                f"Heatmap for the predicted class (**{target_class}**) · "
                f"Model: {MODEL_NAME} · Target layer: `{analysis['layer_name']}`"
            )

            g1, g2, g3 = st.columns(3)
            with g1:
                show_image(image, caption="Original MRI")
            with g2:
                show_image(colored, caption="Grad-CAM heatmap")
            with g3:
                show_image(overlay, caption="Overlay")

            st.caption(
                "Warmer colours (red/yellow) mark regions that pushed the model towards "
                "its prediction. This is **not** a tumour segmentation mask, and the "
                "highlighted area is not guaranteed to be the true lesion."
            )

            st.download_button(
                "⬇️ Download overlay (PNG)",
                data=to_png_bytes(overlay),
                file_name="gradcam_overlay.png",
                mime="image/png",
            )

# =====================================================
# Tab 2: How it works
# =====================================================
with tab_how:
    st.subheader("From MRI slice to prediction")
    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("1 · Preprocess", "The image is converted to RGB and resized to 224 × 224 pixels."),
        ("2 · Extract features", f"{MODEL_NAME}, pre-trained on ImageNet, extracts visual features from the MRI."),
        ("3 · Classify", "A sigmoid output gives the probability that the scan is Tumourous."),
        ("4 · Explain", "Grad-CAM highlights image regions that influenced the prediction."),
    ]
    for col, (title, text) in zip((s1, s2, s3, s4), steps):
        with col:
            st.markdown(
                f'<div class="step-card"><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown(
        """
**Decision threshold.** The model outputs a probability. By default, anything at or
above 0.50 is labelled *Tumourous*. In a screening-style problem, missing a tumour is
usually worse than a false alarm, so a lower threshold (the sidebar slider) trades some
precision for higher recall.

**Grad-CAM.** Gradients of the predicted class are propagated back to the last
convolutional layer. Averaging them gives importance weights for each feature map, and
the weighted sum is shown as a heatmap over the original image.
        """
    )

    metrics = load_metrics()
    if metrics:
        st.subheader("Test-set performance")
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.replace("_", " ").title(), f"{value * 100:.2f}%")

# =====================================================
# Tab 3: About
# =====================================================
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
This app classifies brain MRI slices as **Tumourous** or **Non-Tumourous** using
transfer learning ({MODEL_NAME}). It was trained on a public brain-tumour MRI dataset
in which glioma, meningioma and pituitary tumour images form the *Tumourous* class.

**Limitations**
- Binary only: it does not identify the tumour type or location.
- Trained on a single public dataset; performance on scans from other scanners,
  hospitals or imaging protocols is unknown.
- The dataset contains augmented images, so reported scores may be optimistic.
- Grad-CAM shows model attention, not a verified lesion boundary.
- It should not be used for diagnosis, screening or any clinical decision.
        """
    )

st.markdown(
    '<div class="footer">Built with Streamlit & TensorFlow · For research and education only</div>',
    unsafe_allow_html=True,
)
